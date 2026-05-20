"""Tests für `retrieval.rank_fusion.reciprocal_rank_fusion`."""

from __future__ import annotations

from retrieval.rank_fusion import reciprocal_rank_fusion


def _row(chunk_id: int, content: str = "") -> dict:
    return {"id": chunk_id, "content": content or f"Chunk {chunk_id}"}


def test_rrf_sortiert_nach_summierten_reziproken_raengen():
    """Ein Chunk, der in beiden Listen ganz oben steht, gewinnt gegen
    einen Chunk, der nur in einer Liste ganz oben ist.
    """
    list_a = [_row(1), _row(2), _row(3)]
    list_b = [_row(2), _row(1), _row(3)]

    fused = reciprocal_rank_fusion([list_a, list_b], k=60)
    ids = [r["id"] for r in fused]

    # Chunks 1 und 2 haben beide zwei Top-2-Ränge; Chunk 3 ist schlechter
    assert ids[-1] == 3
    assert set(ids[:2]) == {1, 2}
    # jeder Chunk bekommt einen rrf_score
    assert all("rrf_score" in r for r in fused)


def test_rrf_dedupliziert_identische_chunk_ids():
    """Wenn ein Chunk in mehreren Listen auftaucht, darf er am Ende nur
    einmal im Ergebnis erscheinen.
    """
    list_a = [_row(1), _row(2)]
    list_b = [_row(1), _row(1), _row(3)]  # Chunk 1 sogar doppelt

    fused = reciprocal_rank_fusion([list_a, list_b], k=60)
    ids = [r["id"] for r in fused]
    assert ids.count(1) == 1
    assert sorted(ids) == [1, 2, 3]
