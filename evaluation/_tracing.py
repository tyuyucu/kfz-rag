"""sqlite-basiertes tracing fuer evaluations-laeufe
jeder schritt (ingestion retrieval-stufen generator judge)
schreibt einen eintrag in pipeline_traces
db liegt standardmaessig in evaluation/results/traces.db
"""

from __future__ import annotations

import csv
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Iterable

_DB_LOCK = Lock()
_DEFAULT_DB = Path(__file__).resolve().parent / "results" / "traces.db"


SCHEMA = """
CREATE TABLE IF NOT EXISTS pipeline_traces (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    step_name TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    latency_ms REAL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost_usd REAL,
    question_id TEXT,
    config_name TEXT,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_run_id ON pipeline_traces (run_id);
CREATE INDEX IF NOT EXISTS idx_question_id ON pipeline_traces (question_id);
"""


def _db_path() -> Path:
    path = Path(os.getenv("EVAL_TRACE_DB", _DEFAULT_DB))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def _connect():
    with _DB_LOCK:
        conn = sqlite3.connect(_db_path())
        try:
            conn.executescript(SCHEMA)
            yield conn
            conn.commit()
        finally:
            conn.close()


def new_run_id() -> str:
    """eindeutige run-id im format YYYYMMDD_HHMMSS_<shortuuid>"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{uuid.uuid4().hex[:6]}"


def log_trace(
    run_id: str,
    step_name: str,
    *,
    latency_ms: float | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cost_usd: float | None = None,
    question_id: str | None = None,
    config_name: str | None = None,
    error: str | None = None,
) -> None:
    """schreibt einen schritt-eintrag in die trace-db"""
    with _connect() as conn:
        conn.execute(
            """INSERT INTO pipeline_traces
               (run_id, step_name, timestamp, latency_ms, input_tokens,
                output_tokens, cost_usd, question_id, config_name, error)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                run_id,
                step_name,
                datetime.now().isoformat(timespec="seconds"),
                latency_ms,
                input_tokens,
                output_tokens,
                cost_usd,
                question_id,
                config_name,
                error,
            ),
        )


def get_trace_summary(run_id: str) -> dict[str, Any]:
    """aggregiert die traces eines runs
    gesamtlatenz tokens kosten fehlerquote schritt-weise mittelwerte
    """
    with _connect() as conn:
        cur = conn.execute(
            """SELECT step_name,
                      COUNT(*) AS n,
                      AVG(latency_ms) AS avg_latency,
                      SUM(latency_ms) AS total_latency,
                      SUM(input_tokens) AS in_tok,
                      SUM(output_tokens) AS out_tok,
                      SUM(cost_usd) AS cost,
                      SUM(CASE WHEN error IS NOT NULL THEN 1 ELSE 0 END) AS errors
                 FROM pipeline_traces
                WHERE run_id = ?
                GROUP BY step_name""",
            (run_id,),
        )
        by_step = {}
        totals = {"latency_ms": 0.0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "errors": 0, "rows": 0}
        for row in cur.fetchall():
            step, n, avg_lat, total_lat, in_tok, out_tok, cost, errs = row
            by_step[step] = {
                "count": n,
                "avg_latency_ms": avg_lat,
                "total_latency_ms": total_lat,
                "input_tokens": in_tok or 0,
                "output_tokens": out_tok or 0,
                "cost_usd": cost or 0.0,
                "errors": errs or 0,
            }
            totals["latency_ms"] += total_lat or 0.0
            totals["input_tokens"] += in_tok or 0
            totals["output_tokens"] += out_tok or 0
            totals["cost_usd"] += cost or 0.0
            totals["errors"] += errs or 0
            totals["rows"] += n

    return {"run_id": run_id, "by_step": by_step, "totals": totals}


def export_traces_to_csv(run_id: str, path: str | Path) -> Path:
    """schreibt alle traces eines runs in eine csv"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with _connect() as conn:
        cur = conn.execute(
            """SELECT run_id, step_name, timestamp, latency_ms,
                      input_tokens, output_tokens, cost_usd,
                      question_id, config_name, error
                 FROM pipeline_traces
                WHERE run_id = ?
                ORDER BY id""",
            (run_id,),
        )
        rows: Iterable = cur.fetchall()

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "run_id", "step_name", "timestamp", "latency_ms",
            "input_tokens", "output_tokens", "cost_usd",
            "question_id", "config_name", "error",
        ])
        w.writerows(rows)

    return path
