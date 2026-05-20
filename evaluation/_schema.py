"""helper zum anlegen und befuellen von eval-schemas
fuer ablation und sensitivity laeufe mit abweichender chunk_size
wird ein separates postgres-schema benutzt (z.b. eval_chunk500)
die public-tabellen bleiben dadurch unberuehrt
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path

from config import EMBEDDING_DIMENSION
from db.database import get_connection, release_connection, use_schema

logger = logging.getLogger(__name__)


def schema_name_for_chunk_size(chunk_size: int, chunk_overlap: int = 200) -> str:
    """kanonischer schema-name fuer eine (chunk_size chunk_overlap)-kombi
    bei overlap=200 entfaellt das suffix
    damit alte ablation-schemas weiter wiederverwendet werden
    """
    if chunk_overlap == 200:
        return f"eval_chunk{chunk_size}"
    return f"eval_chunk{chunk_size}_ov{chunk_overlap}"


def ensure_eval_schema(schema: str) -> None:
    """erzeugt schema tabellen und indizes falls noch nicht da"""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(f'CREATE SCHEMA IF NOT EXISTS "{schema}"')
        cur.execute(f'SET search_path TO "{schema}", public')
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                filename TEXT UNIQUE NOT NULL,
                file_hash TEXT NOT NULL,
                page_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id SERIAL PRIMARY KEY,
                document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                page_number INTEGER,
                embedding vector({EMBEDDING_DIMENSION}),
                tsv tsvector GENERATED ALWAYS AS (to_tsvector('german', content)) STORED
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_embedding
            ON chunks USING hnsw (embedding vector_cosine_ops)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_tsv
            ON chunks USING gin(tsv)
        """)

        conn.commit()
        cur.close()
    finally:
        release_connection(conn)


def schema_is_populated(schema: str) -> bool:
    """prueft ob das schema mindestens ein dokument und einen chunk hat"""
    try:
        with use_schema(schema):
            conn = get_connection()
            try:
                cur = conn.cursor()
                cur.execute(
                    'SELECT (SELECT COUNT(*) FROM documents), (SELECT COUNT(*) FROM chunks)'
                )
                doc_count, chunk_count = cur.fetchone()
                cur.close()
                return doc_count > 0 and chunk_count > 0
            finally:
                conn.close()
    except Exception:
        return False


def reingest_into_schema(schema: str, *, chunk_size: int, chunk_overlap: int) -> dict:
    """loescht bestehende daten im schema und ingestet die pdfs neu
    nutzt dieselbe ingestion-pipeline nur die chunker-parameter werden temporaer ueberschrieben
    """
    from ingestion import pipeline as ingestion_pipeline
    from ingestion import chunker as chunker_module

    # parameter-override auf modulebene (synchroner kontext also unkritisch)
    original_size = getattr(chunker_module, "CHUNK_SIZE", None)
    original_overlap = getattr(chunker_module, "CHUNK_OVERLAP", None)
    try:
        chunker_module.CHUNK_SIZE = chunk_size
        chunker_module.CHUNK_OVERLAP = chunk_overlap

        with use_schema(schema):
            ensure_eval_schema(schema)
            # bestehende daten loeschen
            conn = get_connection()
            try:
                cur = conn.cursor()
                cur.execute("TRUNCATE chunks, documents RESTART IDENTITY CASCADE")
                conn.commit()
                cur.close()
            finally:
                conn.close()

            result = ingestion_pipeline.ingest_all_documents()
        return result
    finally:
        if original_size is not None:
            chunker_module.CHUNK_SIZE = original_size
        if original_overlap is not None:
            chunker_module.CHUNK_OVERLAP = original_overlap


@contextmanager
def eval_schema_for_chunk_size(chunk_size: int, *, chunk_overlap: int = 200, reingest_if_empty: bool = True):
    """contextmanager: queries gehen ins eval-schema fuer die (chunk_size chunk_overlap)-kombi
    re-ingest falls das schema noch keine daten hat
    """
    schema = schema_name_for_chunk_size(chunk_size, chunk_overlap)
    ensure_eval_schema(schema)
    if reingest_if_empty and not schema_is_populated(schema):
        logger.info(
            "schema %s leer — starte re-ingest (chunk_size=%s overlap=%s)",
            schema, chunk_size, chunk_overlap,
        )
        reingest_into_schema(schema, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    with use_schema(schema):
        yield schema
