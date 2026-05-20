import logging
from retrieval.multi_query import generate_query_variants
from retrieval.hybrid_search import hybrid_search_query
from retrieval.rank_fusion import reciprocal_rank_fusion
from retrieval.reranker import rerank
from config import RERANK_TOP_K, HYBRID_SEARCH_TOP_K, RERANK_CANDIDATE_POOL

logger = logging.getLogger(__name__)


def retrieve(
    query: str,
    *,
    use_multi_query: bool = True,
    use_hybrid: bool = True,
    use_reranker: bool = True,
    top_k: int | None = None,
    hybrid_top_k: int | None = None,
) -> list[dict]:
    """vollstaendige retrieval-pipeline mit konfigurierbaren stufen
    default: multi-query -> hybrid-search -> rrf -> reranker -> top-k

    use_multi_query: llm-query-varianten verwenden
    use_hybrid: semantik plus volltext per rrf fusionieren
    use_reranker: cross-encoder auf RERANK_CANDIDATE_POOL anwenden
    top_k: finale anzahl chunks (default RERANK_TOP_K)
    hybrid_top_k: kandidaten je teilsuche (default HYBRID_SEARCH_TOP_K)
    """
    top_k = top_k if top_k is not None else RERANK_TOP_K
    hybrid_top_k = hybrid_top_k if hybrid_top_k is not None else HYBRID_SEARCH_TOP_K

    queries = _build_query_list(query, use_multi_query=use_multi_query)

    all_result_lists = _run_search(queries, use_hybrid=use_hybrid, hybrid_top_k=hybrid_top_k)
    if not all_result_lists:
        return []

    fused_results = reciprocal_rank_fusion(all_result_lists)
    if not fused_results:
        return []

    if use_reranker:
        candidates = fused_results[:RERANK_CANDIDATE_POOL]
        try:
            return rerank(query, candidates, top_k=top_k)
        except Exception as e:
            logger.warning("Reranking fehlgeschlagen, nutze RRF-Ergebnisse: %s", e)
            return candidates[:top_k]

    return fused_results[:top_k]


def _build_query_list(query: str, *, use_multi_query: bool) -> list[str]:
    if use_multi_query:
        try:
            return generate_query_variants(query)
        except Exception as e:
            logger.warning("Multi-Query fehlgeschlagen, nutze Originalfrage: %s", e)
            return [query]
    return [query]


def _run_search(
    queries: list[str], *, use_hybrid: bool, hybrid_top_k: int
) -> list[list[dict]]:
    all_result_lists: list[list[dict]] = []
    for q in queries:
        try:
            search_results = hybrid_search_query(q, top_k=hybrid_top_k)
        except Exception as e:
            logger.warning("Suche fehlgeschlagen für Query '%s': %s", q[:50], e)
            continue

        all_result_lists.append(search_results["semantic"])
        if use_hybrid:
            all_result_lists.append(search_results["fulltext"])
    return all_result_lists
