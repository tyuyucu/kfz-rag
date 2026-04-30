import psycopg2
from psycopg2.extras import RealDictCursor
from config import DATABASE_URL, EMBEDDING_DIMENSION


def get_connection():
    return psycopg2.connect(DATABASE_URL)


def init_db():
    """legt alle nötigen tabellen und erweiterungen an."""
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
