"""tests fuer retrieval.hybrid_search.hybrid_search_query
isoliert von der echten db und dem openai-client via mocks
"""

from __future__ import annotations

import pytest

from retrieval import hybrid_search as hybrid_module


def _row(cid: int) -> dict:
    return {"id": cid, "content": f"Chunk {cid}", "score": 0.5}


def test_hybrid_search_kombiniert_beide_suchpfade(monkeypatch):
    called = {"embed": 0, "sem": 0, "ft": 0}

    def fake_embed(q: str):
        called["embed"] += 1
        return [0.0] * 4

    def fake_semantic(embedding, top_k):
        called["sem"] += 1
        return [_row(1), _row(2)]

    def fake_fulltext(q, top_k):
        called["ft"] += 1
        return [_row(3), _row(4)]

    monkeypatch.setattr(hybrid_module, "embed_query", fake_embed)
    monkeypatch.setattr(hybrid_module, "semantic_search", fake_semantic)
    monkeypatch.setattr(hybrid_module, "fulltext_search", fake_fulltext)

    out = hybrid_module.hybrid_search_query("Frage?", top_k=2)
    assert called == {"embed": 1, "sem": 1, "ft": 1}
    assert [r["id"] for r in out["semantic"]] == [1, 2]
    assert [r["id"] for r in out["fulltext"]] == [3, 4]


def test_hybrid_search_ergebnislisten_sind_unabhaengig(monkeypatch):
    """semantic- und fulltext-listen duerfen sich ueberlappen
    bleiben aber zwei getrennte listen
    """
    monkeypatch.setattr(hybrid_module, "embed_query", lambda q: [0.0])
    monkeypatch.setattr(hybrid_module, "semantic_search", lambda e, top_k: [_row(1)])
    monkeypatch.setattr(hybrid_module, "fulltext_search", lambda q, top_k: [_row(1), _row(2)])

    out = hybrid_module.hybrid_search_query("x")
    assert out["semantic"] != out["fulltext"]
    assert out["semantic"][0]["id"] == 1
    assert [r["id"] for r in out["fulltext"]] == [1, 2]
