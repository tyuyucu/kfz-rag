import contextvars
from contextlib import contextmanager

import psycopg2
from psycopg2.extras import RealDictCursor
from config import DATABASE_URL, EMBEDDING_DIMENSION


_current_schema: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_current_schema", default=None
)


def get_connection():
    """Liefert eine Postgres-Verbindung. Wenn ein Eval-Schema aktiv ist
    (siehe `use_schema`), wird `search_path` auf dieses Schema gesetzt —
    damit operiert dieselbe Codebasis wahlweise auf Produktions- oder
    Evaluations-Tabellen, ohne Query-Änderungen.
    """
    conn = psycopg2.connect(DATABASE_URL)
    schema = _current_schema.get()
    if schema:
        cur = conn.cursor()
        cur.execute(f'SET search_path TO "{schema}", public')
        conn.commit()
        cur.close()
    return conn


@contextmanager
def use_schema(name: str | None):
    """Setzt für den gegebenen Block das aktive Schema für alle
    `get_connection()`-Aufrufe. `None` bedeutet Default-Verhalten (public).
    Der Kontext ist ContextVar-basiert und damit pro Thread/Task sicher.
    """
    token = _current_schema.set(name)
    try:
        yield
    finally:
        _current_schema.reset(token)


def init_db():
    """Erstellt alle benötigten Tabellen und Extensions."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                filename TEXT UNIQUE NOT NULL,
                file_hash TEXT NOT NULL,
                page_count INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id SERIAL PRIMARY KEY,
                document_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                page_number INTEGER,
                embedding vector({EMBEDDING_DIMENSION}),
                tsv tsvector GENERATED ALWAYS AS (
                    to_tsvector('german', content)
                ) STORED
            );
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_embedding
            ON chunks USING hnsw (embedding vector_cosine_ops);
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_tsv
            ON chunks USING gin(tsv);
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id SERIAL PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id SERIAL PRIMARY KEY,
                session_id INTEGER REFERENCES chat_sessions(id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Persistente App-Konfiguration (lokale Distribution: Embedding-
        # Provider-Wahl, OpenAI-Embedding-Key, Setup-Status). Daten leben
        # mit der DB im Docker-Volume — Backup-Strategie wie für Vektoren.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS app_config (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        conn.commit()
        cur.close()
    finally:
        conn.close()


# ── Config-Tabelle: Key/Value-Persistenz für lokale App-Einstellungen ──

def get_config(key: str) -> str | None:
    """Liest einen Config-Wert. None, wenn nicht gesetzt."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT value FROM app_config WHERE key = %s", (key,))
        row = cur.fetchone()
        cur.close()
        return row[0] if row else None
    finally:
        conn.close()


def set_config(key: str, value: str | None) -> None:
    """Setzt einen Config-Wert (UPSERT). value=None löscht den Eintrag."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        if value is None:
            cur.execute("DELETE FROM app_config WHERE key = %s", (key,))
        else:
            cur.execute(
                """INSERT INTO app_config (key, value, updated_at)
                   VALUES (%s, %s, CURRENT_TIMESTAMP)
                   ON CONFLICT (key) DO UPDATE SET
                       value = EXCLUDED.value,
                       updated_at = CURRENT_TIMESTAMP""",
                (key, value),
            )
        conn.commit()
        cur.close()
    finally:
        conn.close()


def get_all_config() -> dict[str, str]:
    """Liefert alle Config-Einträge als Dict — z. B. für Debug-Anzeige."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT key, value FROM app_config")
        rows = cur.fetchall()
        cur.close()
        return {k: v for k, v in rows}
    finally:
        conn.close()


def reset_knowledge_base(*, drop_app_config: bool = True) -> None:
    """Setzt die Wissensbasis komplett zurück: löscht alle Chunks,
    Dokumente und Chat-Historie. Wenn `drop_app_config=True` (Default),
    wird auch die Embedding-Provider-Wahl entfernt — der Setup-Wizard
    läuft beim nächsten App-Start erneut.

    Hochgeladene PDF-Dateien im Filesystem-Verzeichnis `documents/`
    bleiben unberührt; sie werden separat gehandhabt, damit der Studi
    sie ggf. ohne erneutes Hochladen wiederverwenden kann.
    """
    conn = get_connection()
    try:
        cur = conn.cursor()
        # CASCADE auf chunks via documents → chunks. Chat-Historie löschen
        # wir explizit, weil sie nicht an documents hängt.
        cur.execute("DELETE FROM chunks;")
        cur.execute("DELETE FROM documents;")
        cur.execute("DELETE FROM chat_messages;")
        cur.execute("DELETE FROM chat_sessions;")
        if drop_app_config:
            cur.execute("DELETE FROM app_config;")
        conn.commit()
        cur.close()
    finally:
        conn.close()


def get_document_by_filename(filename: str) -> dict | None:
    conn = get_connection()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM documents WHERE filename = %s", (filename,))
        row = cur.fetchone()
        cur.close()
        return dict(row) if row else None
    finally:
        conn.close()


def get_all_documents() -> list[dict]:
    conn = get_connection()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM documents ORDER BY created_at DESC")
        rows = cur.fetchall()
        cur.close()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def insert_document(filename: str, file_hash: str, page_count: int) -> int:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO documents (filename, file_hash, page_count) VALUES (%s, %s, %s) RETURNING id",
            (filename, file_hash, page_count)
        )
        doc_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        return doc_id
    finally:
        conn.close()


def delete_document(doc_id: int):
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM documents WHERE id = %s", (doc_id,))
        conn.commit()
        cur.close()
    finally:
        conn.close()


def create_chat_session() -> int:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("INSERT INTO chat_sessions DEFAULT VALUES RETURNING id")
        session_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        return session_id
    finally:
        conn.close()


def save_chat_message(session_id: int, role: str, content: str):
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO chat_messages (session_id, role, content) VALUES (%s, %s, %s)",
            (session_id, role, content)
        )
        conn.commit()
        cur.close()
    finally:
        conn.close()


def get_chat_history(session_id: int, limit: int = 10) -> list[dict]:
    conn = get_connection()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute(
            """SELECT role, content FROM chat_messages
               WHERE session_id = %s ORDER BY created_at DESC LIMIT %s""",
            (session_id, limit)
        )
        rows = cur.fetchall()
        cur.close()
        return list(reversed([dict(r) for r in rows]))
    finally:
        conn.close()
