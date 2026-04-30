from config import RRF_K


def reciprocal_rank_fusion(result_lists: list[list[dict]], k: int = RRF_K) -> list[dict]:
    """führt mehrere ergebnislisten zu einer liste zusammen.

    der rrf-score summiert anteile je nach platzierung in jeder liste.
    """
    fused_scores: dict[int, float] = {}
    chunk_data: dict[int, dict] = {}

    for results in result_lists:
        for rank, result in enumerate(results):
            chunk_id = result["id"]
            if chunk_id not in chunk_data:
                chunk_data[chunk_id] = result

            if chunk_id not in fused_scores:
                fused_scores[chunk_id] = 0.0

            fused_scores[chunk_id] += 1.0 / (k + rank + 1)

    sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)

    fused_results = []
    for chunk_id in sorted_ids:
        result = chunk_data[chunk_id].copy()
        result["rrf_score"] = fused_scores[chunk_id]
        fused_results.append(result)

    return fused_results
