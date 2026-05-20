import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import ContextVar
from dataclasses import dataclass, field

from retrieval.multi_query import generate_query_variants
from retrieval.hybrid_search import hybrid_search_query
from retrieval.rank_fusion import reciprocal_rank_fusion
from retrieval.reranker import rerank
from config import RERANK_TOP_K, HYBRID_SEARCH_TOP_K, RERANK_CANDIDATE_POOL

logger = logging.getLogger(__name__)


# ── latenz-profiling pro retrieve-aufruf ──

@dataclass
class RetrievalTimings:
    """sammelt die latenzen der einzelnen pipeline-stufen pro aufruf"""
    multi_query_ms: float = 0.0
    search_ms: float = 0.0
    fusion_ms: float = 0.0
    rerank_ms: float = 0.0
    total_ms: float = 0.0
    num_queries: int = 0
    num_fused: int = 0
    parallel: bool = False
    details: list[dict] = field(default_factory=list)


# context-var damit der aufrufer (z.b. app.py) die timings nach retrieve abholen kann
_last_timings: ContextVar[RetrievalTimings | None] = ContextVar("_last_timings", default=None)


def get_last_timings() -> RetrievalTimings | None:
    """liefert die latenz-aufstellung des letzten retrieve-aufrufs im aktuellen kontext"""
    return _last_timings.get()


def retrieve(
    query: str,
    *,
    use_multi_query: bool = True,
    use_hybrid: bool = True,
    use_reranker: bool = True,
    top_k: int | None = None,
    hybrid_top_k: int | None = None,
    parallel: bool = True,
) -> list[dict]:
    """vollstaendige retrieval-pipeline mit konfigurierbaren stufen
    default: multi-query -> hybrid-search -> rrf -> reranker -> top-k

    use_multi_query: llm-query-varianten verwenden
    use_hybrid: semantik plus volltext per rrf fusionieren
    use_reranker: cross-encoder auf RERANK_CANDIDATE_POOL anwenden
    top_k: finale anzahl chunks (default RERANK_TOP_K)
    hybrid_top_k: kandidaten je teilsuche (default HYBRID_SEARCH_TOP_K)
    parallel: bei multi-query die suchen parallel ausfuehren
    """
    top_k = top_k if top_k is not None else RERANK_TOP_K
    hybrid_top_k = hybrid_top_k if hybrid_top_k is not None else HYBRID_SEARCH_TOP_K

    timings = RetrievalTimings(parallel=parallel)
    t_start = time.perf_counter()

    # multi-query
    t0 = time.perf_counter()
    queries = _build_query_list(query, use_multi_query=use_multi_query)
    timings.multi_query_ms = (time.perf_counter() - t0) * 1000
    timings.num_queries = len(queries)

    # suche (sequenziell oder parallel)
    t0 = time.perf_counter()
    all_result_lists = _run_search(
        queries, use_hybrid=use_hybrid, hybrid_top_k=hybrid_top_k, parallel=parallel
    )
    timings.search_ms = (time.perf_counter() - t0) * 1000

    if not all_result_lists:
        timings.total_ms = (time.perf_counter() - t_start) * 1000
        _last_timings.set(timings)
        return []

    # rrf
    t0 = time.perf_counter()
    fused_results = reciprocal_rank_fusion(all_result_lists)
    timings.fusion_ms = (time.perf_counter() - t0) * 1000
    timings.num_fused = len(fused_results)

    if not fused_results:
        timings.total_ms = (time.perf_counter() - t_start) * 1000
        _last_timings.set(timings)
        return []

    # reranker
    if use_reranker:
        candidates = fused_results[:RERANK_CANDIDATE_POOL]
        t0 = time.perf_counter()
        try:
            result = rerank(query, candidates, top_k=top_k)
            timings.rerank_ms = (time.perf_counter() - t0) * 1000
        except Exception as e:
            logger.warning("Reranking fehlgeschlagen, nutze RRF-Ergebnisse: %s", e)
            timings.rerank_ms = (time.perf_counter() - t0) * 1000
            result = candidates[:top_k]
        timings.total_ms = (time.perf_counter() - t_start) * 1000
        _last_timings.set(timings)
        return result

    timings.total_ms = (time.perf_counter() - t_start) * 1000
    _last_timings.set(timings)
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
    queries: list[str],
    *,
    use_hybrid: bool,
    hybrid_top_k: int,
    parallel: bool,
) -> list[list[dict]]:
    """fuehrt die hybrid-suche pro query aus
    bei parallel=True und mehr als einer query laufen die calls in einem thread-pool
    """
    if not parallel or len(queries) <= 1:
        return _run_search_sequential(queries, use_hybrid=use_hybrid, hybrid_top_k=hybrid_top_k)
    return _run_search_parallel(queries, use_hybrid=use_hybrid, hybrid_top_k=hybrid_top_k)


def _run_search_sequential(
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


def _run_search_parallel(
    queries: list[str], *, use_hybrid: bool, hybrid_top_k: int
) -> list[list[dict]]:
    """parallel-variante mit thread-pool
    jeder thread holt sich eine eigene db-connection aus dem pool
    """
    # max 4 parallele suchen reicht fuer typische multi-query-counts
    workers = min(len(queries), 4)
    all_result_lists: list[list[dict]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_query = {
            executor.submit(hybrid_search_query, q, top_k=hybrid_top_k): q
            for q in queries
        }
        for fut in as_completed(future_to_query):
            q = future_to_query[fut]
            try:
                search_results = fut.result()
            except Exception as e:
                logger.warning("Suche fehlgeschlagen für Query '%s': %s", q[:50], e)
                continue
            all_result_lists.append(search_results["semantic"])
            if use_hybrid:
                all_result_lists.append(search_results["fulltext"])
    return all_result_lists
