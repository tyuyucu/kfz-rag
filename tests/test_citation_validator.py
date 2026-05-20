"""tests fuer utils.citation_validator"""

from __future__ import annotations

import pytest

from utils.citation_validator import (
    extract_citations,
    validate_citations,
)


CHUNK_VVG = {
    "content": "Nach § 7 Abs. 1 StVG haftet der Halter. Daneben regelt § 115 VVG den Direktanspruch."
}
CHUNK_AKB = {"content": "Die Klausel A.1.5.1 AKB schließt vorsätzliche Schäden aus."}
CHUNK_BGH = {"content": "BGH VI ZR 253/13 (brennendes Kfz) bestätigt die Haftung."}


def test_extract_paragraphen_akb_und_urteile():
    text = (
        "Nach § 7 Abs. 1 StVG und § 115 VVG gilt die Haftung. "
        "Siehe auch A.1.5.1 AKB und D.2.3 AKB. "
        "Das Urteil BGH VI ZR 253/13 bzw. VersR 2014, 1182 sind einschlägig."
    )
    cites = extract_citations(text)
    # nicht alle varianten muessen exakt erfasst werden
    # wir pruefen die wichtigsten klassen
    joined = " || ".join(cites)
    assert "§ 7 Abs. 1 StVG" in joined
    assert "§ 115 VVG" in joined
    assert "A.1.5.1 AKB" in joined
    assert "D.2.3 AKB" in joined
    assert "BGH" in joined and "253/13" in joined
    assert "VersR 2014" in joined


def test_validate_citations_alle_verifiziert():
    answer = "Nach § 115 VVG und A.1.5.1 AKB gilt ..."
    r = validate_citations(answer, source_chunks=[CHUNK_VVG, CHUNK_AKB])
    assert r.citation_precision == 1.0
    assert r.hallucination_risk == "low"
    assert not r.unverified_references


def test_validate_citations_halluziniert_erkennt_risiko():
    answer = "Nach § 999 VVG und BGH VI ZR 123/99 gilt ..."
    r = validate_citations(answer, source_chunks=[CHUNK_VVG])
    assert r.citation_precision == 0.0
    assert r.hallucination_risk == "high"
    assert set(r.unverified_references) == {"§ 999 VVG", "BGH VI ZR 123/99"}


def test_validate_citations_keine_zitate_ist_unkritisch():
    """antworten ohne zitationen sind nicht automatisch halluziniert
    sie koennen auch korrekt sein
    """
    r = validate_citations("Ja, das ist korrekt.", source_chunks=[CHUNK_VVG])
    assert r.citation_precision == 1.0
    assert r.hallucination_risk == "low"
    assert r.cited_references == []
