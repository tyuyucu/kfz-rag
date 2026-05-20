"""Einmal-Helfer: lädt vorhandene Pipeline-Samples (JSON) und führt nur
die RAGAS-Bewertung durch. Vermeidet doppelte API-Kosten, wenn die
Pipeline bereits durchgelaufen ist.

Nutzung:
    python -m evaluation._eval_from_samples <samples.json> [--config-name NAME]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from statistics import mean, median

import config  # noqa: F401  # lädt .env in os.environ
from evaluation._runner import RESULTS_DIR

JUDGE_MODEL = os.getenv("RAGAS_JUDGE_MODEL", "gpt-4o")
GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "gpt-4o-mini")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("samples", help="Pfad zur JSON-Samples-Datei")
    parser.add_argument("--config-name", default="baseline")
    args = parser.parse_args()

    samples_path = Path(args.samples)
    samples = json.loads(samples_path.read_text(encoding="utf-8"))
    samples = [s for s in samples if not s.get("error") and s.get("answer") and s.get("retrieved_contexts")]
    print(f"Geladene Samples: {len(samples)}")

    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper

    rows = []
    for s in samples:
        rows.append({
            "question": s["frage"],
            "answer": s["answer"],
            "contexts": s["retrieved_contexts"],
            "ground_truth": s["referenz_antwort"],
            "reference": s["referenz_antwort"],
            "_id": s["question_id"],
            "_fragetyp": s["fragetyp"],
            "_schwierigkeit": s["schwierigkeit"],
            "_thema": s["thema"],
        })
    dataset = Dataset.from_list(rows)

    judge = LangchainLLMWrapper(ChatOpenAI(model=JUDGE_MODEL, temperature=0.0))
    embs = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

    print(f"RAGAS-Bewertung läuft (Judge: {JUDGE_MODEL}) ...")
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=judge,
        embeddings=embs,
    )
    df = result.to_pandas()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = RESULTS_DIR / f"ragas_{ts}_{args.config_name}_from_samples"
    df.to_csv(base.with_suffix(".scores.csv"), index=False, encoding="utf-8")

    # Summary
    metric_cols = [m for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"] if m in df.columns]
    summary: dict = {
        "run_type": "eval_from_samples",
        "samples_source": str(samples_path),
        "config_name": args.config_name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "judge_model": JUDGE_MODEL,
        "generator_model": GENERATOR_MODEL,
        "n_samples": len(samples),
        "metrics_overall": {},
        "metrics_by_fragetyp": {},
        "metrics_by_schwierigkeit": {},
        "metrics_by_thema": {},
    }

    def group_stats(col):
        out = {}
        if col not in df.columns:
            return out
        for g, sub in df.groupby(col):
            out[str(g)] = {}
            for m in metric_cols:
                vals = [v for v in sub[m].tolist() if v is not None and not (isinstance(v, float) and v != v)]
                if vals:
                    out[str(g)][m] = {"mean": float(mean(vals)), "median": float(median(vals)), "n": len(vals)}
        return out

    for m in metric_cols:
        vals = [v for v in df[m].tolist() if v is not None and not (isinstance(v, float) and v != v)]
        if vals:
            summary["metrics_overall"][m] = {
                "mean": float(mean(vals)),
                "median": float(median(vals)),
                "min": float(min(vals)),
                "max": float(max(vals)),
                "n": len(vals),
            }

    summary["metrics_by_fragetyp"] = group_stats("_fragetyp")
    summary["metrics_by_schwierigkeit"] = group_stats("_schwierigkeit")
    summary["metrics_by_thema"] = group_stats("_thema")

    base.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    md = [f"# RAGAS-Zusammenfassung — {args.config_name} (aus Samples {samples_path.name})\n"]
    md.append(f"- Generator: `{GENERATOR_MODEL}`, Judge: `{JUDGE_MODEL}`")
    md.append(f"- N Samples: {len(samples)}")
    md.append("")
    md.append("## Gesamt-Metriken\n")
    md.append("| Metrik | Mean | Median | Min | Max | n |")
    md.append("|--------|-----:|-------:|----:|----:|--:|")
    for m, s in summary["metrics_overall"].items():
        md.append(f"| {m} | {s['mean']:.3f} | {s['median']:.3f} | {s['min']:.3f} | {s['max']:.3f} | {s['n']} |")
    md.append("")
    for label, key in [("Nach Fragetyp", "metrics_by_fragetyp"), ("Nach Schwierigkeit", "metrics_by_schwierigkeit"), ("Nach Thema", "metrics_by_thema")]:
        if not summary[key]:
            continue
        md.append(f"## {label}\n")
        md.append("| Gruppe | Metrik | Mean | Median | n |")
        md.append("|--------|--------|-----:|-------:|--:|")
        for g, ms in sorted(summary[key].items()):
            for m, s in ms.items():
                md.append(f"| {g} | {m} | {s['mean']:.3f} | {s['median']:.3f} | {s['n']} |")
        md.append("")

    base.with_suffix(".summary.md").write_text("\n".join(md), encoding="utf-8")

    print(f"→ {base.with_suffix('.scores.csv').name}")
    print(f"→ {base.with_suffix('.summary.md').name}")
    print(f"→ {base.with_suffix('.summary.json').name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
