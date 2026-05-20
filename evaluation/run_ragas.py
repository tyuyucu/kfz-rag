"""RAGAS-Evaluation für den Kfz-RAG-Testkatalog.

Metriken (RAGAS v0.1+):
- faithfulness: Wie gut stützt sich die Antwort auf die abgerufenen Kontexte?
- answer_relevancy: Beantwortet die Antwort die Frage?
- context_precision: Wie relevant sind die abgerufenen Kontexte?
- context_recall: Deckt der abgerufene Kontext die Referenzantwort ab?
  (wird nur bewertet, wenn `referenz_antwort` vorhanden ist)

Judge ≠ Generator:
Um Self-Preference-Bias (Zheng et al. 2023, arXiv:2306.05685) zu vermeiden,
wird der Generator (`gpt-4o-mini`) von einem stärkeren Judge-Modell
(`gpt-4o`) bewertet. Das Judge-Modell lässt sich über die Umgebungs-
variable RAGAS_JUDGE_MODEL überschreiben (Default: gpt-4o).

Temperature = 0 für reproduzierbare Ergebnisse.

Nutzung:
    python -m evaluation.run_ragas [--dry-run] [--limit N] [--config-name NAME]

Ausgabe:
    evaluation/results/ragas_<timestamp>.csv
    evaluation/results/ragas_<timestamp>_summary.md
    evaluation/results/ragas_<timestamp>_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

# Lokale Imports
from evaluation._cost import estimate_run_cost
from evaluation._runner import (
    EvalSample,
    RESULTS_DIR,
    filter_questions,
    load_testkatalog,
    run_pipeline_for_question,
    write_samples_csv,
    write_samples_json,
)
from evaluation._tracing import export_traces_to_csv, get_trace_summary, new_run_id
from retrieval.pipeline import retrieve
from generation.generator import generate_answer_with_usage


JUDGE_MODEL = os.getenv("RAGAS_JUDGE_MODEL", "gpt-4o")
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "gpt-4o-mini")


def _eval_generate(query: str, chunks: list[dict]):
    """Generierung mit temperature=0.0 für deterministische Evaluation."""
    return generate_answer_with_usage(query, chunks, temperature=0.0)


def _build_ragas_dataset(samples: list[EvalSample]):
    """Baut das RAGAS-Dataset aus den EvalSamples."""
    from datasets import Dataset

    rows = []
    for s in samples:
        if s.error or not s.retrieved_contexts:
            continue
        rows.append({
            "question": s.frage,
            "answer": s.answer,
            "contexts": s.retrieved_contexts,
            "ground_truth": s.referenz_antwort,
            "reference": s.referenz_antwort,
            # Custom Felder für Gruppierung im Summary
            "_id": s.question_id,
            "_fragetyp": s.fragetyp,
            "_schwierigkeit": s.schwierigkeit,
            "_thema": s.thema,
        })
    return Dataset.from_list(rows)


def _evaluate_with_ragas(dataset, *, judge_model: str):
    """Ruft ragas.evaluate auf und liefert ein DataFrame zurück.

    Konfiguration des Judge-Modells über langchain_openai.ChatOpenAI.
    """
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
    embeddings = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small")
    )

    metrics = [faithfulness, answer_relevancy, context_precision]
    # context_recall braucht ground_truth — bei uns immer vorhanden
    metrics.append(context_recall)

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=embeddings,
    )
    return result.to_pandas()


def _group_stats(df, group_col: str, metric_cols: list[str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    if group_col not in df.columns:
        return out
    for group, sub in df.groupby(group_col):
        out[str(group)] = {}
        for m in metric_cols:
            if m not in sub.columns:
                continue
            vals = [v for v in sub[m].tolist() if v is not None and not _is_nan(v)]
            if not vals:
                continue
            out[str(group)][m] = {
                "mean": mean(vals),
                "median": median(vals),
                "min": min(vals),
                "max": max(vals),
                "n": len(vals),
            }
    return out


def _is_nan(x) -> bool:
    import math
    return isinstance(x, float) and math.isnan(x)


def _write_summary(
    df,
    *,
    run_id: str,
    config_name: str,
    samples: list[EvalSample],
    base_path: Path,
    metric_cols: list[str],
) -> tuple[Path, Path]:
    """Schreibt Markdown- und JSON-Zusammenfassung."""
    summary: dict[str, Any] = {
        "run_id": run_id,
        "config_name": config_name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "judge_model": JUDGE_MODEL,
        "generator_model": GENERATOR_MODEL,
        "n_questions": len(samples),
        "n_successful": int(sum(1 for s in samples if not s.error)),
        "latency": {
            "avg_retrieval_ms": mean([s.retrieval_latency_ms for s in samples if s.retrieval_latency_ms]) if samples else 0.0,
            "avg_generation_ms": mean([s.generation_latency_ms for s in samples if s.generation_latency_ms]) if samples else 0.0,
            "avg_total_ms": mean([s.total_latency_ms for s in samples if s.total_latency_ms]) if samples else 0.0,
        },
        "cost_usd": {
            "generator_total": sum(s.generator_cost_usd for s in samples),
        },
        "metrics_overall": {},
        "metrics_by_fragetyp": _group_stats(df, "_fragetyp", metric_cols),
        "metrics_by_schwierigkeit": _group_stats(df, "_schwierigkeit", metric_cols),
        "metrics_by_thema": _group_stats(df, "_thema", metric_cols),
    }

    for m in metric_cols:
        if m not in df.columns:
            continue
        vals = [v for v in df[m].tolist() if v is not None and not _is_nan(v)]
        if not vals:
            continue
        summary["metrics_overall"][m] = {
            "mean": mean(vals),
            "median": median(vals),
            "min": min(vals),
            "max": max(vals),
            "n": len(vals),
        }

    json_path = base_path.with_suffix(".summary.json")
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    md_lines: list[str] = []
    md_lines.append(f"# RAGAS-Zusammenfassung — Run `{run_id}`")
    md_lines.append("")
    md_lines.append(f"- **Konfiguration:** `{config_name}`")
    md_lines.append(f"- **Generator:** `{GENERATOR_MODEL}`, **Judge:** `{JUDGE_MODEL}`")
    md_lines.append(f"- **Fragen:** {summary['n_questions']} (davon {summary['n_successful']} erfolgreich)")
    md_lines.append(f"- **Avg. Latenz:** Retrieval {summary['latency']['avg_retrieval_ms']:.0f} ms · Generation {summary['latency']['avg_generation_ms']:.0f} ms · Gesamt {summary['latency']['avg_total_ms']:.0f} ms")
    md_lines.append(f"- **Generator-Kosten:** ${summary['cost_usd']['generator_total']:.4f}")
    md_lines.append("")
    md_lines.append("## Gesamt-Metriken")
    md_lines.append("")
    md_lines.append("| Metrik | Mean | Median | Min | Max | n |")
    md_lines.append("|--------|-----:|-------:|----:|----:|--:|")
    for m, stats in summary["metrics_overall"].items():
        md_lines.append(
            f"| {m} | {stats['mean']:.3f} | {stats['median']:.3f} | {stats['min']:.3f} | {stats['max']:.3f} | {stats['n']} |"
        )
    md_lines.append("")

    for group_key, title in [
        ("metrics_by_fragetyp", "Nach Fragetyp"),
        ("metrics_by_schwierigkeit", "Nach Schwierigkeit"),
        ("metrics_by_thema", "Nach Thema"),
    ]:
        if not summary[group_key]:
            continue
        md_lines.append(f"## {title}")
        md_lines.append("")
        md_lines.append("| Gruppe | Metrik | Mean | Median | n |")
        md_lines.append("|--------|--------|-----:|-------:|--:|")
        for g, ms in sorted(summary[group_key].items()):
            for m, stats in ms.items():
                md_lines.append(
                    f"| {g} | {m} | {stats['mean']:.3f} | {stats['median']:.3f} | {stats['n']} |"
                )
        md_lines.append("")

    md_path = base_path.with_suffix(".summary.md")
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    return json_path, md_path


def main() -> int:
    parser = argparse.ArgumentParser(description="RAGAS-Evaluation des Kfz-RAG-Systems")
    parser.add_argument("--dry-run", action="store_true", help="Nur 3 Fragen durchlaufen (Wiring-Test)")
    parser.add_argument("--limit", type=int, default=None, help="Maximalanzahl Fragen")
    parser.add_argument("--config-name", default="default", help="Kennzeichen der aktuellen Konfiguration")
    parser.add_argument("--no-multi-query", action="store_true")
    parser.add_argument("--no-hybrid", action="store_true")
    parser.add_argument("--no-reranker", action="store_true")
    parser.add_argument("--yes", action="store_true", help="Kostenschätzung ohne Bestätigung akzeptieren")
    args = parser.parse_args()

    questions = load_testkatalog()
    if args.dry_run:
        questions = filter_questions(questions, limit=3)
    elif args.limit:
        questions = filter_questions(questions, limit=args.limit)

    n = len(questions)
    if n == 0:
        print("Keine Fragen gefunden — ist der Testkatalog leer?", file=sys.stderr)
        return 2

    est = estimate_run_cost(n, generator_model=GENERATOR_MODEL, judge_model=JUDGE_MODEL)
    print("==================================================")
    print(f"RAGAS-Evaluation — {n} Fragen, Konfig: {args.config_name}")
    print(f"Generator: {GENERATOR_MODEL} · Judge: {JUDGE_MODEL}")
    print(f"Geschätzte Kosten: Generator ${est['generator_usd']:.4f} + Judge ${est['judge_usd']:.4f}")
    print(f"                   + Embedding ${est['embedding_usd']:.4f} = ${est['total_usd']:.4f}")
    print("==================================================")

    if not args.yes and not args.dry_run:
        answer = input("Fortfahren? [y/N] ").strip().lower()
        if answer not in ("y", "yes", "j", "ja"):
            print("Abgebrochen.")
            return 1

    retrieve_kwargs = {
        "use_multi_query": not args.no_multi_query,
        "use_hybrid": not args.no_hybrid,
        "use_reranker": not args.no_reranker,
    }

    run_id = new_run_id()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"ragas_{ts}_{args.config_name}"
    csv_path = RESULTS_DIR / f"{base_name}.csv"
    json_path = RESULTS_DIR / f"{base_name}.json"
    traces_path = RESULTS_DIR / f"{base_name}_traces.csv"

    try:
        from tqdm import tqdm
        iterator = tqdm(questions, desc="Pipeline", unit="q")
    except ImportError:
        iterator = questions

    samples: list[EvalSample] = []
    for q in iterator:
        s = run_pipeline_for_question(
            q,
            run_id=run_id,
            config_name=args.config_name,
            retrieve_fn=retrieve,
            generate_fn=_eval_generate,
            retrieve_kwargs=retrieve_kwargs,
            generator_model=GENERATOR_MODEL,
        )
        samples.append(s)

    write_samples_csv(samples, csv_path)
    write_samples_json(samples, json_path)
    export_traces_to_csv(run_id, traces_path)
    print(f"\n→ Samples: {csv_path.name}, {json_path.name}")
    print(f"→ Traces: {traces_path.name}")

    # RAGAS-Metriken nur berechnen, wenn Samples brauchbar sind
    successful = [s for s in samples if not s.error and s.answer and s.retrieved_contexts]
    if not successful:
        print("⚠ Keine erfolgreichen Samples — RAGAS-Bewertung übersprungen.")
        return 1
    if args.dry_run:
        print("→ --dry-run aktiv: RAGAS-Bewertung wird übersprungen (Wiring-Test).")
        trace_summary = get_trace_summary(run_id)
        print(f"→ Trace-Summary: {trace_summary['totals']}")
        return 0

    print(f"\n▶ RAGAS-Bewertung läuft (Judge: {JUDGE_MODEL}) ...")
    dataset = _build_ragas_dataset(successful)
    df = _evaluate_with_ragas(dataset, judge_model=JUDGE_MODEL)

    per_question_csv = RESULTS_DIR / f"{base_name}_scores.csv"
    df.to_csv(per_question_csv, index=False, encoding="utf-8")

    metric_cols = [c for c in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"] if c in df.columns]
    j_path, m_path = _write_summary(
        df,
        run_id=run_id,
        config_name=args.config_name,
        samples=samples,
        base_path=RESULTS_DIR / base_name,
        metric_cols=metric_cols,
    )
    print(f"→ Scores: {per_question_csv.name}")
    print(f"→ Summary: {m_path.name}, {j_path.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
