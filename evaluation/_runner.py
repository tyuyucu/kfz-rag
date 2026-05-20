"""runner-infrastruktur fuer die evaluations-skripte
- laedt den testkatalog
- orchestriert retrieval und generierung pro frage
- speichert traces (siehe _tracing.py)
- export-helfer fuer csv/json/markdown
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from config import LLM_MODEL

from ._cost import cost_usd
from ._timing import reset as reset_measurements, drain_measurements
from ._tracing import log_trace

EVAL_DIR = Path(__file__).resolve().parent
TESTKATALOG_PATH = EVAL_DIR / "testkatalog.json"
RESULTS_DIR = EVAL_DIR / "results"


@dataclass
class EvalSample:
    """ergebnis eines einzelnen frage-durchlaufs"""
    question_id: str
    frage: str
    referenz_antwort: str
    referenz_kontext: str
    quelle_dokument: str
    fragetyp: str
    schwierigkeit: str
    thema: str
    retrieved_contexts: list[str] = field(default_factory=list)
    retrieved_sources: list[dict[str, Any]] = field(default_factory=list)
    answer: str = ""
    retrieval_latency_ms: float = 0.0
    generation_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    generator_input_tokens: int = 0
    generator_output_tokens: int = 0
    generator_cost_usd: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_testkatalog(path: Path = TESTKATALOG_PATH) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["fragen"]


def filter_questions(
    questions: list[dict[str, Any]],
    *,
    fragetyp: str | None = None,
    schwierigkeit: str | None = None,
    thema: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    out = questions
    if fragetyp:
        out = [q for q in out if q["fragetyp"] == fragetyp]
    if schwierigkeit:
        out = [q for q in out if q["schwierigkeit"] == schwierigkeit]
    if thema:
        out = [q for q in out if q["thema"] == thema]
    if limit is not None:
        out = out[:limit]
    return out


def run_pipeline_for_question(
    question: dict[str, Any],
    *,
    run_id: str,
    config_name: str,
    retrieve_fn,
    generate_fn,
    retrieve_kwargs: dict[str, Any] | None = None,
    generator_model: str = LLM_MODEL,
) -> EvalSample:
    """fuehrt retrieval und generierung fuer eine frage aus und logt traces
    retrieve_fn(query **kwargs) -> list[dict] mit text und metadaten
    generate_fn(query chunks) -> (answer_text usage_dict|None)
    """
    retrieve_kwargs = retrieve_kwargs or {}
    sample = EvalSample(
        question_id=question["id"],
        frage=question["frage"],
        referenz_antwort=question["referenz_antwort"],
        referenz_kontext=question["referenz_kontext"],
        quelle_dokument=question["quelle_dokument"],
        fragetyp=question["fragetyp"],
        schwierigkeit=question["schwierigkeit"],
        thema=question["thema"],
    )

    reset_measurements()
    t_total = time.perf_counter()

    try:
        t0 = time.perf_counter()
        chunks = retrieve_fn(question["frage"], **retrieve_kwargs)
        sample.retrieval_latency_ms = (time.perf_counter() - t0) * 1000
        sample.retrieved_contexts = [c.get("content") or c.get("text", "") for c in chunks]
        sample.retrieved_sources = [
            {
                "filename": c.get("filename"),
                "page_number": c.get("page_number"),
                "chunk_index": c.get("chunk_index"),
                "score": c.get("rerank_score") or c.get("rrf_score") or c.get("score"),
            }
            for c in chunks
        ]
        log_trace(
            run_id,
            "retrieval",
            latency_ms=sample.retrieval_latency_ms,
            question_id=question["id"],
            config_name=config_name,
        )
    except Exception as e:
        sample.error = f"retrieval: {e}"
        log_trace(
            run_id,
            "retrieval",
            question_id=question["id"],
            config_name=config_name,
            error=str(e),
        )
        return sample

    try:
        t0 = time.perf_counter()
        answer, usage = generate_fn(question["frage"], chunks)
        sample.generation_latency_ms = (time.perf_counter() - t0) * 1000
        sample.answer = answer or ""
        if usage:
            sample.generator_input_tokens = usage.get("prompt_tokens", 0)
            sample.generator_output_tokens = usage.get("completion_tokens", 0)
            sample.generator_cost_usd = cost_usd(
                generator_model,
                sample.generator_input_tokens,
                sample.generator_output_tokens,
            )
        log_trace(
            run_id,
            "generation",
            latency_ms=sample.generation_latency_ms,
            input_tokens=sample.generator_input_tokens,
            output_tokens=sample.generator_output_tokens,
            cost_usd=sample.generator_cost_usd,
            question_id=question["id"],
            config_name=config_name,
        )
    except Exception as e:
        sample.error = f"generation: {e}"
        log_trace(
            run_id,
            "generation",
            question_id=question["id"],
            config_name=config_name,
            error=str(e),
        )

    sample.total_latency_ms = (time.perf_counter() - t_total) * 1000
    drain_measurements()
    return sample


def write_samples_csv(samples: list[EvalSample], path: Path) -> Path:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "question_id", "thema", "fragetyp", "schwierigkeit",
        "frage", "answer", "referenz_antwort",
        "retrieval_latency_ms", "generation_latency_ms", "total_latency_ms",
        "generator_input_tokens", "generator_output_tokens", "generator_cost_usd",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in samples:
            row = {k: s.to_dict().get(k) for k in fields}
            w.writerow(row)
    return path


def write_samples_json(samples: list[EvalSample], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([s.to_dict() for s in samples], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path
