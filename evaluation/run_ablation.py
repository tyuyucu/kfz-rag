"""Ablation-Harness für den Kfz-RAG-Stack.

Durchläuft 7 vordefinierte Konfigurationen und misst Faithfulness,
Answer Relevancy, Context Precision, Context Recall, Latenz und Kosten
je Konfiguration. Die Deltas gegenüber `baseline` werden tabellarisch
und in drei Matplotlib-PNGs dargestellt.

Konfigurationen mit abweichender Chunk-Größe nutzen ein separates
Postgres-Schema (siehe `evaluation/_schema.py`), damit die
Produktions-Tabellen unberührt bleiben.

Nutzung:
    python -m evaluation.run_ablation [--dry-run] [--limit N] [--yes]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
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
    write_samples_csv,
)
from evaluation._schema import eval_schema_for_chunk_size
from evaluation._tracing import new_run_id
from retrieval.pipeline import retrieve
from generation.generator import generate_answer_with_usage

JUDGE_MODEL = os.getenv("RAGAS_JUDGE_MODEL", "gpt-4o")
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "gpt-4o-mini")


@dataclass
class AblationConfig:
    name: str
    retrieve_kwargs: dict[str, Any]
    chunk_size: int = 1000
    chunk_overlap: int = 200
    description: str = ""


CONFIGS: list[AblationConfig] = [
    AblationConfig(
        name="baseline",
        retrieve_kwargs={"use_multi_query": False, "use_hybrid": True, "use_reranker": False},
        description="Hybrid-Suche (semantisch + Volltext) + RRF, ohne Multi-Query, ohne Reranker",
    ),
    AblationConfig(
        name="small_chunks",
        retrieve_kwargs={"use_multi_query": True, "use_hybrid": True, "use_reranker": True},
        chunk_size=500,
        description="Volle Pipeline mit chunk_size=500",
    ),
    AblationConfig(
        name="large_chunks",
        retrieve_kwargs={"use_multi_query": True, "use_hybrid": True, "use_reranker": True},
        chunk_size=2000,
        description="Volle Pipeline mit chunk_size=2000",
    ),
    AblationConfig(
        name="multi_query",
        retrieve_kwargs={"use_multi_query": True, "use_hybrid": True, "use_reranker": False},
        description="Baseline + Multi-Query (ohne Reranker)",
    ),
    AblationConfig(
        name="reranker",
        retrieve_kwargs={"use_multi_query": False, "use_hybrid": True, "use_reranker": True},
        description="Baseline + Cross-Encoder-Reranker (ohne Multi-Query)",
    ),
    AblationConfig(
        name="multi_query_plus_reranker",
        retrieve_kwargs={"use_multi_query": True, "use_hybrid": True, "use_reranker": True},
        description="Volle Pipeline: Multi-Query + Hybrid + Reranker",
    ),
]


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
            "_id": s.question_id,
            "_thema": s.thema,
        })
    return Dataset.from_list(rows)


def _evaluate_ragas(dataset, *, judge_model: str):
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    judge_llm = LangchainLLMWrapper(ChatOpenAI(model=judge_model, temperature=0.0))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge_llm,
        embeddings=embeddings,
    )
    return result.to_pandas()


def _run_config(
    config: AblationConfig,
    questions: list[dict[str, Any]],
    *,
    run_id: str,
    generator_model: str,
) -> tuple[list[EvalSample], dict[str, float]]:
    """Läuft eine einzelne Konfiguration. Liefert Samples + Metriken-Mittelwerte."""
    samples: list[EvalSample] = []

    def _generate(query, chunks):
        return generate_answer_with_usage(query, chunks, temperature=0.0)

    def _do_run():
        try:
            from tqdm import tqdm
            it = tqdm(questions, desc=f"  {config.name}", unit="q", leave=False)
        except ImportError:
            it = questions
        for q in it:
            s = run_pipeline_for_question(
                q,
                run_id=run_id,
                config_name=config.name,
                retrieve_fn=retrieve,
                generate_fn=_generate,
                retrieve_kwargs=config.retrieve_kwargs,
                generator_model=generator_model,
            )
            samples.append(s)

    if config.chunk_size != 1000:
        with eval_schema_for_chunk_size(
            config.chunk_size, chunk_overlap=config.chunk_overlap
        ):
            _do_run()
    else:
        _do_run()

    # Aggregate
    successful = [s for s in samples if not s.error and s.answer]
    metrics = {
        "n": len(samples),
        "n_successful": len(successful),
        "avg_retrieval_ms": mean([s.retrieval_latency_ms for s in samples if s.retrieval_latency_ms]) if samples else 0.0,
        "avg_generation_ms": mean([s.generation_latency_ms for s in samples if s.generation_latency_ms]) if samples else 0.0,
        "avg_total_ms": mean([s.total_latency_ms for s in samples if s.total_latency_ms]) if samples else 0.0,
        "total_cost_usd": sum(s.generator_cost_usd for s in samples),
    }
    return samples, metrics


def _compute_ragas_for_config(samples: list[EvalSample], *, judge_model: str) -> dict[str, float]:
    import math

    dataset = _build_ragas_dataset(samples)
    if len(dataset) == 0:
        return {}
    df = _evaluate_ragas(dataset, judge_model=judge_model)
    out = {}
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        if m in df.columns:
            # NaN-Werte (z. B. aus RAGAS-Judge-Timeouts) explizit ausfiltern
            vals = [
                v for v in df[m].tolist()
                if v is not None and not (isinstance(v, float) and math.isnan(v))
            ]
            if vals:
                out[m] = float(mean(vals))
                out[f"{m}_n"] = len(vals)
    return out


def _write_deltas_md(
    results: dict[str, dict[str, float]], *, run_id: str, path: Path
) -> None:
    baseline = results.get("baseline", {})
    lines: list[str] = []
    lines.append(f"# Ablation-Ergebnisse — Run `{run_id}`")
    lines.append("")
    lines.append("Deltas werden relativ zur Konfiguration `baseline` berechnet.")
    lines.append("")
    metric_cols = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    header = "| Konfiguration | " + " | ".join(metric_cols) + " | Latenz ∅ (ms) | Kosten (USD) |"
    sep = "|---" * (len(metric_cols) + 3) + "|"
    lines.append(header)
    lines.append(sep)
    for cfg_name, metrics in results.items():
        row = [cfg_name]
        for m in metric_cols:
            val = metrics.get(m)
            if val is None:
                row.append("–")
            else:
                if cfg_name == "baseline" or m not in baseline:
                    row.append(f"{val:.3f}")
                else:
                    delta = val - baseline[m]
                    row.append(f"{val:.3f} ({delta:+.3f})")
        row.append(f"{metrics.get('avg_total_ms', 0):.0f}")
        row.append(f"{metrics.get('total_cost_usd', 0):.4f}")
        lines.append("| " + " | ".join(row) + " |")
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot(results: dict[str, dict[str, float]], *, base_path: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    configs = list(results.keys())
    out: list[Path] = []

    # 1. Faithfulness
    fig, ax = plt.subplots(figsize=(10, 5))
    vals = [results[c].get("faithfulness", 0.0) for c in configs]
    ax.bar(configs, vals, color="#6366f1")
    ax.set_title("Faithfulness je Konfiguration")
    ax.set_ylabel("Faithfulness")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    p = base_path.with_name(base_path.name + "_faithfulness.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)
    out.append(p)

    # 2. Context Precision
    fig, ax = plt.subplots(figsize=(10, 5))
    vals = [results[c].get("context_precision", 0.0) for c in configs]
    ax.bar(configs, vals, color="#10b981")
    ax.set_title("Context Precision je Konfiguration")
    ax.set_ylabel("Context Precision")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    p = base_path.with_name(base_path.name + "_context_precision.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)
    out.append(p)

    # 3. Latency vs Faithfulness scatter
    fig, ax = plt.subplots(figsize=(8, 6))
    for c in configs:
        x = results[c].get("avg_total_ms", 0.0)
        y = results[c].get("faithfulness", 0.0)
        ax.scatter(x, y, s=120, alpha=0.8)
        ax.annotate(c, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=9)
    ax.set_xlabel("Avg. Gesamtlatenz (ms)")
    ax.set_ylabel("Faithfulness")
    ax.set_title("Latenz vs. Faithfulness")
    fig.tight_layout()
    p = base_path.with_name(base_path.name + "_latency_vs_faithfulness.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)
    out.append(p)

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Ablation-Harness für kfz-rag")
    parser.add_argument("--dry-run", action="store_true", help="Nur 3 Fragen je Konfiguration")
    parser.add_argument("--limit", type=int, default=None, help="Maximalanzahl Fragen")
    parser.add_argument("--skip-chunk-configs", action="store_true", help="Chunk-Size-Varianten überspringen")
    parser.add_argument("--only", type=str, default=None,
                        help="Nur diese Konfigurationen laufen lassen (kommasepariert, z.B. 'reranker,multi_query_plus_reranker')")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--skip-ragas", action="store_true", help="Nur Pipeline-Latenz, keine RAGAS-Bewertung")
    args = parser.parse_args()

    questions = load_testkatalog()
    if args.dry_run:
        questions = filter_questions(questions, limit=3)
    elif args.limit:
        questions = filter_questions(questions, limit=args.limit)

    configs_to_run = [
        c for c in CONFIGS if not (args.skip_chunk_configs and c.chunk_size != 1000)
    ]
    if args.only:
        wanted = {n.strip() for n in args.only.split(",") if n.strip()}
        configs_to_run = [c for c in configs_to_run if c.name in wanted]
        if not configs_to_run:
            print(f"Keine Konfiguration matcht --only={args.only}", file=sys.stderr)
            return 2
    n_questions = len(questions)
    n_configs = len(configs_to_run)

    if n_questions == 0 or n_configs == 0:
        print("Keine Fragen oder keine Konfigurationen zum Ausführen.", file=sys.stderr)
        return 2

    est = estimate_run_cost(
        n_questions * n_configs,
        generator_model=GENERATOR_MODEL,
        judge_model=JUDGE_MODEL,
    )
    print("==================================================")
    print(f"Ablation — {n_configs} Konfigs × {n_questions} Fragen = {n_configs*n_questions} Runs")
    print(f"Generator: {GENERATOR_MODEL} · Judge: {JUDGE_MODEL}")
    print(f"Geschätzte Kosten: ~${est['total_usd']:.4f} (Generator + Judge + Embedding)")
    print("==================================================")

    if not args.yes and not args.dry_run:
        if input("Fortfahren? [y/N] ").strip().lower() not in ("y", "yes", "j", "ja"):
            print("Abgebrochen.")
            return 1

    run_id = new_run_id()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"ablation_{ts}"
    csv_path = RESULTS_DIR / f"{base_name}.csv"
    md_path = RESULTS_DIR / f"{base_name}_deltas.md"

    # Sammelstruktur
    results: dict[str, dict[str, float]] = {}
    all_samples_rows: list[dict[str, Any]] = []

    for cfg in configs_to_run:
        print(f"\n▶ Konfig: {cfg.name} — {cfg.description}")
        samples, pipeline_metrics = _run_config(
            cfg, questions, run_id=run_id, generator_model=GENERATOR_MODEL
        )
        ragas_metrics: dict[str, float] = {}
        if not args.skip_ragas and not args.dry_run:
            print(f"  → RAGAS-Bewertung für {cfg.name} ...")
            try:
                ragas_metrics = _compute_ragas_for_config(samples, judge_model=JUDGE_MODEL)
            except Exception as e:
                print(f"  ⚠ RAGAS für {cfg.name} fehlgeschlagen: {e}", file=sys.stderr)
        results[cfg.name] = {**pipeline_metrics, **ragas_metrics}
        for s in samples:
            row = s.to_dict()
            row["config_name"] = cfg.name
            all_samples_rows.append(row)

    # Export aller Samples als eine CSV
    import csv as _csv
    if all_samples_rows:
        keys = sorted({k for row in all_samples_rows for k in row.keys()})
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = _csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for row in all_samples_rows:
                w.writerow({k: row.get(k) for k in keys})

    _write_deltas_md(results, run_id=run_id, path=md_path)

    # Ergebnis-JSON
    json_path = RESULTS_DIR / f"{base_name}_results.json"
    json_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Plots
    plot_base = RESULTS_DIR / base_name
    try:
        pngs = _plot(results, base_path=plot_base)
        for p in pngs:
            print(f"→ Plot: {p.name}")
    except Exception as e:
        print(f"⚠ Plot-Erzeugung fehlgeschlagen: {e}", file=sys.stderr)

    print(f"\n→ Deltas: {md_path.name}")
    print(f"→ Samples: {csv_path.name}")
    print(f"→ Ergebnisse: {json_path.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
