import logging
from retrieval.multi_query import generate_query_variants
from retrieval.hybrid_search import hybrid_search_query
from retrieval.rank_fusion import reciprocal_rank_fusion
from retrieval.reranker import rerank
from config import RERANK_TOP_K

logger = logging.getLogger(__name__)


def retrieve(query: str, top_k: int = RERANK_TOP_K) -> list[dict]:
    """kompletter abruf-ablauf mit sauberem fallback bei fehlern.

    1. mehrere query-varianten erstellen
    2. pro variante semantisch und per volltext suchen
    3. ergebnisse zusammenführen
    4. beste treffer neu sortieren

    wenn ein schritt fehlschlägt, läuft der rest trotzdem weiter.
    """
    # 1. mehrere query-varianten erzeugen (fallback: nur original-query)
    try:
        queries = generate_query_variants(query)
    except Exception as e:
        logger.warning("Multi-Query fehlgeschlagen, nutze Original-Query: %s", e)
        queries = [query]

    # 2. hybrid-suche für jede query-variante
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

    # 3. ergebnisse zusammenführen
    fused_results = reciprocal_rank_fusion(all_result_lists)

    if not fused_results:
        return []

    # 4. top-ergebnisse neu sortieren (fallback: ohne reranking)
    candidates = fused_results[:20]
    try:
        final_results = rerank(query, candidates, top_k=top_k)
    except Exception as e:
        logger.warning("Reranking fehlgeschlagen, nutze RRF-Ergebnisse: %s", e)
        final_results = candidates[:top_k]

    return final_results
