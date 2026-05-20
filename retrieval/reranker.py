from sentence_transformers import CrossEncoder
from config import RERANKER_MODEL, RERANK_TOP_K

_reranker = None


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def rerank(query: str, results: list[dict], top_k: int = RERANK_TOP_K) -> list[dict]:
    """Bewertet die Ergebnisse mit einem Cross-Encoder neu.

    Der Cross-Encoder berechnet die Relevanz von (Query, Passage)-Paaren
    präziser als die initiale Vektorsuche.
    """
    if not results:
        return []

    model = get_reranker()

    pairs = [(query, r["content"]) for r in results]
    scores = model.predict(pairs)

    for i, result in enumerate(results):
        result["rerank_score"] = float(scores[i])

    reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:top_k]
