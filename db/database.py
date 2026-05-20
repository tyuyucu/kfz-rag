import contextvars
from contextlib import contextmanager

import psycopg2
from psycopg2.extras import RealDictCursor
from config import DATABASE_URL, EMBEDDING_DIMENSION


_current_schema: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "_current_schema", default=None
)


def get_connection():
    """liefert eine postgres-verbindung
    bei aktivem schema-context wird der search_path entsprechend gesetzt
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
    """setzt das aktive schema fuer alle get_connection-aufrufe im block
    None bedeutet default-verhalten (public)
    contextvar-basiert also thread-safe
    """
    token = _current_schema.set(name)
    try:
        yield
    finally:
        _current_schema.reset(token)


def init_db():
    """erstellt alle tabellen und extensions"""
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

        # persistente app-config (key-value-store)
        # haelt openai-embedding-key und setup-status
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


# ── config-tabelle key-value-persistenz ──

def get_config(key: str) -> str | None:
    """liest einen config-wert
    None wenn nicht gesetzt
    """
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
    """setzt einen config-wert (upsert)
    value=None loescht den eintrag
    """
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
    """alle config-eintraege als dict"""
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
    """setzt die wissensbasis komplett zurueck
    loescht chunks dokumente und chat-historie
    drop_app_config=True loescht zusaetzlich die setup-config
    sodass der wizard beim naechsten start wieder laeuft

    pdfs im documents-ordner bleiben unberuehrt
    """
    conn = get_connection()
    try:
        cur = conn.cursor()
        # cascade ueber documents loescht auch die chunks
        # chat-historie haengt nicht an documents also explizit loeschen
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
