import logging
from retrieval.multi_query import generate_query_variants
from retrieval.hybrid_search import hybrid_search_query
from retrieval.rank_fusion import reciprocal_rank_fusion
from retrieval.reranker import rerank
from config import RERANK_TOP_K

logger = logging.getLogger(__name__)


def retrieve(query: str, top_k: int = RERANK_TOP_K) -> list[dict]:
    """Vollstaendige Retrieval-Pipeline mit Fehlerbehandlung.

    1. Multi-Query: Generiert Varianten der Suchanfrage
    2. Hybrid Search: Semantische + Volltext-Suche für jede Variante
    3. Reciprocal Rank Fusion: Kombiniert alle Ergebnisse
    4. Reranking: Cross-Encoder bewertet Top-Ergebnisse neu

    Bei Fehlern in einzelnen Schritten wird mit reduzierten Ergebnissen fortgefahren.
    """
    # 1. Multi-Query Generierung (Fallback: nur Original-Query)
    try:
        queries = generate_query_variants(query)
    except Exception as e:
        logger.warning("Multi-Query fehlgeschlagen, nutze Original-Query: %s", e)
        queries = [query]

    # 2. Hybrid Search für jede Query-Variante
    all_result_lists = []
    for q in queries:
        try:
            search_results = hybrid_search_query(q)
            all_result_lists.append(search_results["semantic"])
            all_result_lists.append(search_results["fulltext"])
        except Exception as e:
            logger.warning("Hybrid Search fehlgeschlagen für Query '%s': %s", q[:50], e)

    if not all_result_lists:
        return []

    # 3. Reciprocal Rank Fusion
    fused_results = reciprocal_rank_fusion(all_result_lists)

    if not fused_results:
        return []

    # 4. Reranking der Top-Ergebnisse (Fallback: ohne Reranking)
    candidates = fused_results[:20]
    try:
        final_results = rerank(query, candidates, top_k=top_k)
    except Exception as e:
        logger.warning("Reranking fehlgeschlagen, nutze RRF-Ergebnisse: %s", e)
        final_results = candidates[:top_k]

    return final_results
