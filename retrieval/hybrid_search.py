from ingestion.embedder import embed_query
from db.vector_store import semantic_search, fulltext_search
from config import HYBRID_SEARCH_TOP_K


def hybrid_search_query(query: str, top_k: int = HYBRID_SEARCH_TOP_K) -> dict:
    """Fuehrt parallele semantische und Volltext-Suche durch.

    Returns: dict mit 'semantic' und 'fulltext' Ergebnislisten
    """
    query_embedding = embed_query(query)
    semantic_results = semantic_search(query_embedding, top_k=top_k)
    fulltext_results = fulltext_search(query, top_k=top_k)

    return {
        "semantic": semantic_results,
        "fulltext": fulltext_results
    }
