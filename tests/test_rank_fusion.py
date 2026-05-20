"""tests fuer retrieval.rank_fusion.reciprocal_rank_fusion"""

from __future__ import annotations

from retrieval.rank_fusion import reciprocal_rank_fusion


def _row(chunk_id: int, content: str = "") -> dict:
    return {"id": chunk_id, "content": content or f"Chunk {chunk_id}"}


def test_rrf_sortiert_nach_summierten_reziproken_raengen():
    """ein chunk der in beiden listen oben ist gewinnt
    gegen einen chunk der nur in einer liste oben ist
    """
    list_a = [_row(1), _row(2), _row(3)]
    list_b = [_row(2), _row(1), _row(3)]

    fused = reciprocal_rank_fusion([list_a, list_b], k=60)
    ids = [r["id"] for r in fused]

    # chunks 1 und 2 haben beide zwei top-2-raenge chunk 3 ist schlechter
    assert ids[-1] == 3
    assert set(ids[:2]) == {1, 2}
    # jeder chunk bekommt einen rrf_score
    assert all("rrf_score" in r for r in fused)


def test_rrf_dedupliziert_identische_chunk_ids():
    """ein chunk der in mehreren listen vorkommt
    darf am ende nur einmal im ergebnis erscheinen
    """
    list_a = [_row(1), _row(2)]
    list_b = [_row(1), _row(1), _row(3)]  # chunk 1 sogar doppelt

    fused = reciprocal_rank_fusion([list_a, list_b], k=60)
    ids = [r["id"] for r in fused]
    assert ids.count(1) == 1
    assert sorted(ids) == [1, 2, 3]
