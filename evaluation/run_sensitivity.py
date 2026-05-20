"""sensitivitaetsanalyse fuer zentrale pipeline-parameter

zwei parametergitter:
1) chunk_size x chunk_overlap (mit re-ingest pro kombi)
2) top_k (kandidaten nach reranking ohne re-ingest)

kombinationen mit overlap > 20% * chunk_size werden uebersprungen

aus kostengruenden nur auf den medium-fragen des testkatalogs (N=20)

ausgaben:
- csv pro kombi mit faithfulness und context_precision
- zwei heatmap-pngs

nutzung:
    python -m evaluation.run_sensitivity [--dry-run] [--yes]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

from evaluation._cost import estimate_run_cost
from evaluation._runner import (
    EvalSample,
    RESULTS_DIR,
    filter_questions,
    load_testkatalog,
    run_pipeline_for_question,
)
from evaluation._schema import eval_schema_for_chunk_size
from evaluation._tracing import new_run_id
from retrieval.pipeline import retrieve
from generation.generator import generate_answer_with_usage

JUDGE_MODEL = os.getenv("RAGAS_JUDGE_MODEL", "gpt-4o")
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "gpt-4o-mini")

CHUNK_SIZES = [500, 750, 1000, 1500, 2000]
OVERLAPS = [0, 100, 200]
TOP_KS = [3, 5, 10, 20]


def _chunk_overlap_valid(chunk_size: int, overlap: int) -> bool:
    """Overlap > 20 % von chunk_size wird verworfen."""
    return overlap <= int(chunk_size * 0.2)


def _build_ragas_dataset(samples: list[EvalSample]):
    from datasets import Dataset
    rows = []
    for s in samples:
        if s.error or not s.retrieved_contexts or not s.answer:
            continue
        rows.append({
            "question": s.frage,
            "answer": s.answer,
            "contexts": s.retrieved_contexts,
            "ground_truth": s.referenz_antwort,
            "reference": s.referenz_antwort,
        })
    return Dataset.from_list(rows)


def _evaluate_ragas(dataset, *, judge_model: str) -> dict[str, float]:
    import math
    from ragas import evaluate
    from ragas.metrics import faithfulness, context_precision
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    judge_llm = LangchainLLMWrapper(ChatOpenAI(model=judge_model, temperature=0.0))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, context_precision],
        llm=judge_llm,
        embeddings=embeddings,
    )
    df = result.to_pandas()
    out = {}
    for m in ["faithfulness", "context_precision"]:
        if m in df.columns:
            # NaN-Werte (RAGAS-Judge-Timeouts) ausfiltern
            vals = [
                v for v in df[m].tolist()
                if v is not None and not (isinstance(v, float) and math.isnan(v))
            ]
            if vals:
                out[m] = float(mean(vals))
                out[f"{m}_n"] = len(vals)
    return out


def _run_pipeline_once(
    questions: list[dict[str, Any]],
    *,
    run_id: str,
    config_name: str,
    retrieve_kwargs: dict[str, Any],
) -> list[EvalSample]:
    def _generate(query, chunks):
        return generate_answer_with_usage(query, chunks, temperature=0.0)

    samples = []
    try:
        from tqdm import tqdm
        it = tqdm(questions, desc=f"  {config_name}", unit="q", leave=False)
    except ImportError:
        it = questions
    for q in it:
        samples.append(run_pipeline_for_question(
            q,
            run_id=run_id,
            config_name=config_name,
            retrieve_fn=retrieve,
            generate_fn=_generate,
            retrieve_kwargs=retrieve_kwargs,
            generator_model=GENERATOR_MODEL,
        ))
    return samples


def _run_chunk_overlap_grid(questions, *, run_id: str, dry_run: bool) -> list[dict[str, Any]]:
    rows = []
    for chunk_size in CHUNK_SIZES:
        for overlap in OVERLAPS:
            if not _chunk_overlap_valid(chunk_size, overlap):
                continue
            cfg_name = f"cs{chunk_size}_ov{overlap}"
            print(f"▶ {cfg_name}")
            try:
                if chunk_size != 1000 or overlap != 200:
                    with eval_schema_for_chunk_size(chunk_size, chunk_overlap=overlap):
                        samples = _run_pipeline_once(
                            questions,
                            run_id=run_id,
                            config_name=cfg_name,
                            retrieve_kwargs={
                                "use_multi_query": True,
                                "use_hybrid": True,
                                "use_reranker": True,
                            },
                        )
                else:
                    samples = _run_pipeline_once(
                        questions,
                        run_id=run_id,
                        config_name=cfg_name,
                        retrieve_kwargs={
                            "use_multi_query": True,
                            "use_hybrid": True,
                            "use_reranker": True,
                        },
                    )
            except Exception as e:
                print(f"  ⚠ Fehler: {e}", file=sys.stderr)
                continue
            metrics = {}
            if not dry_run:
                try:
                    ds = _build_ragas_dataset(samples)
                    if len(ds) > 0:
                        metrics = _evaluate_ragas(ds, judge_model=JUDGE_MODEL)
                except Exception as e:
                    print(f"  ⚠ RAGAS fehlgeschlagen: {e}", file=sys.stderr)
            rows.append({
                "dim": "chunk_overlap",
                "chunk_size": chunk_size,
                "overlap": overlap,
                "top_k": 5,
                "n_samples": len(samples),
                **metrics,
            })
    return rows


def _run_topk_grid(questions, *, run_id: str, dry_run: bool) -> list[dict[str, Any]]:
    rows = []
    for top_k in TOP_KS:
        cfg_name = f"topk{top_k}"
        print(f"▶ {cfg_name}")
        samples = _run_pipeline_once(
            questions,
            run_id=run_id,
            config_name=cfg_name,
            retrieve_kwargs={
                "use_multi_query": True,
                "use_hybrid": True,
                "use_reranker": True,
                "top_k": top_k,
            },
        )
        metrics = {}
        if not dry_run:
            try:
                ds = _build_ragas_dataset(samples)
                if len(ds) > 0:
                    metrics = _evaluate_ragas(ds, judge_model=JUDGE_MODEL)
            except Exception as e:
                print(f"  ⚠ RAGAS fehlgeschlagen: {e}", file=sys.stderr)
        rows.append({
            "dim": "topk",
            "chunk_size": 1000,
            "overlap": 200,
            "top_k": top_k,
            "n_samples": len(samples),
            **metrics,
        })
    return rows


def _plot_heatmaps(rows: list[dict[str, Any]], base_path: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    out: list[Path] = []

    # Heatmap 1: chunk_size × overlap → faithfulness
    co_rows = [r for r in rows if r["dim"] == "chunk_overlap"]
    if co_rows:
        matrix = np.full((len(CHUNK_SIZES), len(OVERLAPS)), np.nan)
        for r in co_rows:
            try:
                i = CHUNK_SIZES.index(r["chunk_size"])
                j = OVERLAPS.index(r["overlap"])
                matrix[i, j] = r.get("faithfulness", np.nan)
            except ValueError:
                continue
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(matrix, cmap="viridis", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(OVERLAPS)))
        ax.set_xticklabels([str(o) for o in OVERLAPS])
        ax.set_yticks(range(len(CHUNK_SIZES)))
        ax.set_yticklabels([str(c) for c in CHUNK_SIZES])
        ax.set_xlabel("Overlap")
        ax.set_ylabel("Chunk-Size")
        ax.set_title("Faithfulness — chunk_size × overlap")
        for i in range(len(CHUNK_SIZES)):
            for j in range(len(OVERLAPS)):
                val = matrix[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white", fontsize=9)
        fig.colorbar(im, ax=ax, label="Faithfulness")
        fig.tight_layout()
        p = base_path.with_name(base_path.name + "_chunk_overlap_heatmap.png")
        fig.savefig(p, dpi=120)
        plt.close(fig)
        out.append(p)

    # Heatmap 2: chunk_size × top_k
    # Für eine separate Matrix: für jede chunk_size + top_k (mit overlap=200)
    topk_rows = [r for r in rows if r["dim"] == "topk"]
    co_rows_default_overlap = [r for r in rows if r["dim"] == "chunk_overlap" and r["overlap"] == 200]
    if topk_rows or co_rows_default_overlap:
        # Matrix zusammenbauen — wir nehmen nur top_k-Variationen bei chunk_size=1000
        matrix2 = np.full((len(CHUNK_SIZES), len(TOP_KS)), np.nan)
        # Default-Linie: bei overlap=200 und top_k=5, aus co_rows_default_overlap
        for r in co_rows_default_overlap:
            try:
                i = CHUNK_SIZES.index(r["chunk_size"])
                j = TOP_KS.index(5)
                matrix2[i, j] = r.get("faithfulness", np.nan)
            except ValueError:
                continue
        # topk-Variationen bei chunk_size=1000
        for r in topk_rows:
            try:
                i = CHUNK_SIZES.index(r["chunk_size"])
                j = TOP_KS.index(r["top_k"])
                matrix2[i, j] = r.get("faithfulness", np.nan)
            except ValueError:
                continue
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(matrix2, cmap="viridis", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(TOP_KS)))
        ax.set_xticklabels([str(k) for k in TOP_KS])
        ax.set_yticks(range(len(CHUNK_SIZES)))
        ax.set_yticklabels([str(c) for c in CHUNK_SIZES])
        ax.set_xlabel("Top-K nach Reranking")
        ax.set_ylabel("Chunk-Size")
        ax.set_title("Faithfulness — chunk_size × top_k (Overlap=200)")
        for i in range(len(CHUNK_SIZES)):
            for j in range(len(TOP_KS)):
                val = matrix2[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="white", fontsize=9)
        fig.colorbar(im, ax=ax, label="Faithfulness")
        fig.tight_layout()
        p = base_path.with_name(base_path.name + "_chunk_topk_heatmap.png")
        fig.savefig(p, dpi=120)
        plt.close(fig)
        out.append(p)

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Sensitivitätsanalyse für kfz-rag")
    parser.add_argument("--dry-run", action="store_true", help="Nur Wiring-Test, keine RAGAS-Bewertung")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--skip-chunk-overlap", action="store_true")
    parser.add_argument("--skip-topk", action="store_true")
    args = parser.parse_args()

    all_questions = load_testkatalog()
    questions = filter_questions(all_questions, schwierigkeit="medium")
    n_q = len(questions)

    # Grobe Run-Anzahl
    co_count = sum(1 for cs in CHUNK_SIZES for ov in OVERLAPS if _chunk_overlap_valid(cs, ov))
    n_runs = (0 if args.skip_chunk_overlap else co_count) + (0 if args.skip_topk else len(TOP_KS))
    total_questions = n_runs * n_q
    est = estimate_run_cost(total_questions, generator_model=GENERATOR_MODEL, judge_model=JUDGE_MODEL)

    print("==================================================")
    print(f"Sensitivität — {n_runs} Konfigurationen × {n_q} Fragen = {total_questions} Runs")
    print(f"Generator: {GENERATOR_MODEL} · Judge: {JUDGE_MODEL}")
    print(f"Geschätzte Kosten: ~${est['total_usd']:.4f}")
    if not args.skip_chunk_overlap:
        print(f"Re-Ingestionen für abweichende chunk_size/overlap werden automatisch angestoßen.")
    print("==================================================")

    if not args.yes and not args.dry_run:
        if input("Fortfahren? [y/N] ").strip().lower() not in ("y", "yes", "j", "ja"):
            print("Abgebrochen.")
            return 1

    run_id = new_run_id()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"sensitivity_{ts}"
    csv_path = RESULTS_DIR / f"{base_name}.csv"

    all_rows: list[dict[str, Any]] = []
    if not args.skip_chunk_overlap:
        all_rows.extend(_run_chunk_overlap_grid(questions, run_id=run_id, dry_run=args.dry_run))
    if not args.skip_topk:
        all_rows.extend(_run_topk_grid(questions, run_id=run_id, dry_run=args.dry_run))

    import csv as _csv
    keys = sorted({k for r in all_rows for k in r.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = _csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r.get(k) for k in keys})

    print(f"\n→ Rohdaten: {csv_path.name}")

    # Zusätzlich als JSON
    (RESULTS_DIR / f"{base_name}.json").write_text(
        json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    try:
        pngs = _plot_heatmaps(all_rows, RESULTS_DIR / base_name)
        for p in pngs:
            print(f"→ Heatmap: {p.name}")
    except Exception as e:
        print(f"⚠ Plot-Erzeugung fehlgeschlagen: {e}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
