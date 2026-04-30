from db.database import get_connection


def insert_chunks(document_id: int, chunks: list[dict]):
    """speichert abschnitte mit embeddings in der datenbank."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        for chunk in chunks:
            embedding_str = "[" + ",".join(str(x) for x in chunk["embedding"]) + "]"
            cur.execute(
                """INSERT INTO chunks (document_id, chunk_index, content, page_number, embedding)
                   VALUES (%s, %s, %s, %s, %s::vector)""",
                (document_id, chunk["chunk_index"], chunk["content"],
                 chunk["page_number"], embedding_str)
            )
        conn.commit()
        cur.close()
    finally:
        conn.close()


def semantic_search(query_embedding: list[float], top_k: int = 20) -> list[dict]:
    """sucht passende abschnitte über semantische ähnlichkeit."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"
        cur.execute(
            """SELECT c.id, c.content, c.page_number, c.chunk_index,
                      d.filename,
                      1 - (c.embedding <=> %s::vector) AS similarity
               FROM chunks c
               JOIN documents d ON c.document_id = d.id
               ORDER BY c.embedding <=> %s::vector
               LIMIT %s""",
            (embedding_str, embedding_str, top_k)
        )
        results = []
        for row in cur.fetchall():
            results.append({
                "id": row[0], "content": row[1], "page_number": row[2],
                "chunk_index": row[3], "filename": row[4], "score": float(row[5])
            })
        cur.close()
        return results
    finally:
        conn.close()


def fulltext_search(query: str, top_k: int = 20) -> list[dict]:
    """macht eine volltextsuche in postgres mit deutschem wörterbuch."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """SELECT c.id, c.content, c.page_number, c.chunk_index,
                      d.filename,
                      ts_rank(c.tsv, plainto_tsquery('german', %s)) AS rank
               FROM chunks c
               JOIN documents d ON c.document_id = d.id
               WHERE c.tsv @@ plainto_tsquery('german', %s)
               ORDER BY rank DESC
               LIMIT %s""",
            (query, query, top_k)
        )
        results = []
        for row in cur.fetchall():
            results.append({
                "id": row[0], "content": row[1], "page_number": row[2],
                "chunk_index": row[3], "filename": row[4], "score": float(row[5])
            })
        cur.close()
        return results
    finally:
        conn.close()


def get_chunk_count() -> int:
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM chunks")
        count = cur.fetchone()[0]
        cur.close()
        return count
    finally:
        conn.close()
