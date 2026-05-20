"""Benchmark: execute_values-Bulk-Insert vs. Schleifen-INSERT.

Misst die reine Insertion-Zeit beider Varianten mit identischen Chunks
(extrahiert aus einem PDF). Schreibt NICHT in die Produktions-`public`-
Tabellen — nutzt stattdessen ein eigenes Schema `bench_bulk_insert`.

Aufruf:
    python -m scripts.bench_bulk_insert [--pdf <name>]

Ergebnis: Zeitmessung beider Varianten + Ausgabe als Markdown-Tabelle
nach `docs/performance_bulk_insert.md` (neue Sektion "Messwerte").
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from psycopg2.extras import execute_values

import config  # lädt .env in os.environ
from db.database import get_connection, use_schema
from evaluation._schema import ensure_eval_schema
from ingestion.chunker import chunk_pages
from ingestion.embedder import embed_texts
from ingestion.pdf_loader import extract_text_from_pdf

BENCH_SCHEMA = "bench_bulk_insert"


def _row_tuple(doc_id: int, chunk: dict) -> tuple:
    return (
        doc_id,
        chunk["chunk_index"],
        chunk["content"],
        chunk["page_number"],
        "[" + ",".join(str(x) for x in chunk["embedding"]) + "]",
    )


def insert_loop(doc_id: int, chunks: list[dict]) -> None:
    """Altmodische Schleifen-Variante (pro Chunk ein INSERT)."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        for c in chunks:
            cur.execute(
                """INSERT INTO chunks
                       (document_id, chunk_index, content, page_number, embedding)
                   VALUES (%s, %s, %s, %s, %s::vector)""",
                _row_tuple(doc_id, c),
            )
        conn.commit()
        cur.close()
    finally:
        conn.close()


def insert_bulk(doc_id: int, chunks: list[dict]) -> None:
    """Neue Bulk-Variante (execute_values)."""
    if not chunks:
        return
    rows = [_row_tuple(doc_id, c) for c in chunks]
    conn = get_connection()
    try:
        cur = conn.cursor()
        execute_values(
            cur,
            """INSERT INTO chunks
                   (document_id, chunk_index, content, page_number, embedding)
               VALUES %s""",
            rows,
            template="(%s, %s, %s, %s, %s::vector)",
            page_size=200,
        )
        conn.commit()
        cur.close()
    finally:
        conn.close()


def _prepare_chunks(pdf_path: Path) -> list[dict]:
    pages = extract_text_from_pdf(str(pdf_path))
    chunks = chunk_pages(pages)
    texts = [c["content"] for c in chunks]
    print(f"[bench] {len(chunks)} Chunks — embedde ...")
    embs = embed_texts(texts)
    for i, c in enumerate(chunks):
        c["embedding"] = embs[i]
    return chunks


def _reset_schema_tables() -> None:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("TRUNCATE chunks, documents RESTART IDENTITY CASCADE")
        conn.commit()
        cur.close()
    finally:
        conn.close()


def _insert_doc_row(filename: str) -> int:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO documents (filename, file_hash, page_count) VALUES (%s, %s, %s) RETURNING id",
            (filename, "bench", 0),
        )
        doc_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        return doc_id
    finally:
        conn.close()


def _append_markdown(result: dict) -> None:
    path = Path(__file__).resolve().parent.parent / "docs" / "performance_bulk_insert.md"
    if not path.exists():
        return
    t = path.read_text(encoding="utf-8")
    marker = "## Messwerte"
    block = [
        "",
        "## Messwerte",
        "",
        f"Benchmark vom {time.strftime('%Y-%m-%d %H:%M')} auf lokaler Docker-Postgres",
        f"(`pgvector/pgvector:pg16`), Schema `{BENCH_SCHEMA}`, gleiche Chunks, "
        f"gleiche Embeddings.",
        "",
        f"- PDF: `{result['pdf']}`",
        f"- Chunks: {result['chunks']}",
        "",
        "| Variante | Zeit (ms) | Chunks/s |",
        "|----------|---------:|---------:|",
        f"| Schleifen-INSERT (alt) | {result['loop_ms']:.0f} | {result['chunks'] / (result['loop_ms']/1000):.1f} |",
        f"| execute_values Bulk    | {result['bulk_ms']:.0f} | {result['chunks'] / (result['bulk_ms']/1000):.1f} |",
        f"| **Faktor Bulk vs. Loop** | **{result['loop_ms']/result['bulk_ms']:.1f} ×** | |",
        "",
    ]
    if marker in t:
        head, _tail = t.split(marker, 1)
        new = head + "\n".join(block).lstrip() + "\n"
    else:
        new = t.rstrip() + "\n\n" + "\n".join(block).lstrip() + "\n"
    path.write_text(new, encoding="utf-8")
    print(f"[bench] Markdown aktualisiert: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", default="Skript_KFZ_2023.pdf")
    args = parser.parse_args()

    pdf_path = Path(os.getenv("DOCUMENTS_DIR", "documents")) / args.pdf
    if not pdf_path.exists():
        print(f"PDF nicht gefunden: {pdf_path}", file=sys.stderr)
        return 1

    # Schema einmalig anlegen
    ensure_eval_schema(BENCH_SCHEMA)

    with use_schema(BENCH_SCHEMA):
        # Chunks+Embeddings einmal vorbereiten — für beide Varianten identisch
        chunks = _prepare_chunks(pdf_path)
        n = len(chunks)

        # Variante A: Loop
        _reset_schema_tables()
        doc_id = _insert_doc_row(args.pdf)
        print("[bench] starte Loop-INSERT ...")
        t0 = time.perf_counter()
        insert_loop(doc_id, chunks)
        loop_ms = (time.perf_counter() - t0) * 1000
        print(f"[bench] Loop-INSERT: {loop_ms:.0f} ms ({n} Chunks)")

        # Variante B: Bulk
        _reset_schema_tables()
        doc_id = _insert_doc_row(args.pdf)
        print("[bench] starte Bulk-INSERT ...")
        t0 = time.perf_counter()
        insert_bulk(doc_id, chunks)
        bulk_ms = (time.perf_counter() - t0) * 1000
        print(f"[bench] Bulk-INSERT: {bulk_ms:.0f} ms ({n} Chunks)")

        # Aufräumen (Schema bleibt, Tabellen leer)
        _reset_schema_tables()

    result = {
        "pdf": args.pdf,
        "chunks": n,
        "loop_ms": loop_ms,
        "bulk_ms": bulk_ms,
    }
    _append_markdown(result)

    print()
    print(f"Speedup: {loop_ms/bulk_ms:.1f}× ({loop_ms:.0f} ms → {bulk_ms:.0f} ms)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
