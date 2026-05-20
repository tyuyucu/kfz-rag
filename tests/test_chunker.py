"""tests fuer ingestion.chunker.chunk_pages"""

from __future__ import annotations

import pytest

from ingestion import chunker as chunker_module
from ingestion.chunker import chunk_pages


def _page(num: int, text: str) -> dict:
    return {"page_number": num, "text": text}


def test_chunker_respektiert_chunk_size(monkeypatch):
    """einzelne chunks ueberschreiten chunk_size + reserve nicht deutlich"""
    monkeypatch.setattr(chunker_module, "CHUNK_SIZE", 200)
    monkeypatch.setattr(chunker_module, "CHUNK_OVERLAP", 20)

    long_text = ". ".join(f"Satz {i}" for i in range(200))
    chunks = chunk_pages([_page(1, long_text)])

    assert len(chunks) > 1
    # splitter ist etwas weicher daher toleranz
    oversized = [c for c in chunks if len(c["content"]) > 260]
    assert not oversized, f"Erwartet alle Chunks ≤ 260 Zeichen, fand: {[len(c['content']) for c in oversized]}"


def test_chunker_overlap_erzeugt_teilweise_shared_content(monkeypatch):
    """bei overlap teilen aufeinanderfolgende chunks ein paar zeichen
    indiz: gesamt-laenge der chunks > originaltext-laenge wenn overlap > 0
    """
    monkeypatch.setattr(chunker_module, "CHUNK_SIZE", 100)
    monkeypatch.setattr(chunker_module, "CHUNK_OVERLAP", 30)

    text = "A" * 500
    chunks = chunk_pages([_page(1, text)])
    total_len = sum(len(c["content"]) for c in chunks)
    assert total_len > len(text), "Erwartet Overlap → Summe > Originaltext"


def test_chunker_leerer_input_liefert_leere_liste():
    assert chunk_pages([]) == []
