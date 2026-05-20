"""Zitationskontrolle für generierte Antworten.

Erkennt juristische Zitate (Paragraphen, AKB-Klauseln, BGH-Urteile) via
Regex und prüft, ob sie in den abgerufenen Kontext-Chunks oder den
referenzierten Quelldokumenten tatsächlich vorkommen. Liefert eine
Einschätzung des Halluzinationsrisikos.

Ziel: frühes Erkennen von *erfundenen* Zitaten — dem klassischen
RAG-Halluzinationsmuster.

Die Funktion ist eigenständig nutzbar (siehe `tests/test_citation_validator.py`)
und kann optional in die Generierung integriert werden (z. B. als Post-
Processing-Schritt in `generation/generator.py`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ──────────────────────────────────────────────
# Regex-Patterns für die drei Zitations-Klassen
# ──────────────────────────────────────────────

_PARAGRAPH_PATTERN = re.compile(
    r"§+\s?\d+\w*"
    r"(?:\s*(?:Abs\.|Absatz)\s?\d+\w*)?"
    r"(?:\s*(?:S\.|Satz)\s?\d+)?"
    r"(?:\s*(?:Nr\.|Nummer)\s?\d+)?"
    r"\s+(?:VVG|StVG|BGB|PflVG|KfzPflVV|StGB|FZV|HGB)",
    flags=re.IGNORECASE,
)

_AKB_PATTERN = re.compile(
    r"\b[A-E]\.\d+(?:\.\d+)*\s*AKB\b",
)

_BGH_PATTERN = re.compile(
    r"(?:BGH|OLG\s+\w+|LG\s+\w+)\s*"
    r"(?:Urt\.?(?:\s+v\.?\s*\d{1,2}\.\d{1,2}\.\d{4})?\s*-\s*)?"
    r"(?:VI\s+)?ZR\s*\d+/\d+",
)

_FUNDSTELLE_PATTERN = re.compile(
    r"(?:VersR|NJW|NZV|NVwZ|r\+s)\s*\d{4},?\s*\d+",
)


@dataclass
class CitationValidationResult:
    """Ergebnis einer Zitations-Prüfung."""

    cited_references: list[str] = field(default_factory=list)
    verified_references: list[str] = field(default_factory=list)
    unverified_references: list[str] = field(default_factory=list)
    citation_precision: float = 0.0
    hallucination_risk: str = "low"  # low / medium / high

    def to_dict(self) -> dict:
        return {
            "cited_references": list(self.cited_references),
            "verified_references": list(self.verified_references),
            "unverified_references": list(self.unverified_references),
            "citation_precision": round(self.citation_precision, 3),
            "hallucination_risk": self.hallucination_risk,
        }


def extract_citations(text: str) -> list[str]:
    """Extrahiert alle erkennbaren Zitate aus einem Text.

    Mehrfach-Treffer werden nicht dedupliziert — so sieht man, welche
    Zitate überproportional oft vorkommen.
    """
    if not text:
        return []
    matches: list[str] = []
    for pat in (_PARAGRAPH_PATTERN, _AKB_PATTERN, _BGH_PATTERN, _FUNDSTELLE_PATTERN):
        for m in pat.finditer(text):
            matches.append(_normalize_citation(m.group(0)))
    return matches


def _normalize_citation(ref: str) -> str:
    """Normalisiert Whitespace und einheitliches `§`-Symbol."""
    ref = re.sub(r"\s+", " ", ref).strip()
    ref = ref.replace("§§", "§")
    return ref


def _is_citation_in_text(citation: str, haystack: str) -> bool:
    """Tolerante Suche: normalisiert beide Seiten und akzeptiert kleine
    Abweichungen (z. B. fehlendes Leerzeichen um `§`).
    """
    norm_haystack = re.sub(r"\s+", " ", haystack)
    norm_cite = re.sub(r"\s+", " ", citation)
    # tolerante Variante: entferne Leerzeichen um Zahlen
    def strip_ws(s: str) -> str:
        return re.sub(r"\s+", "", s.lower())

    return strip_ws(norm_cite) in strip_ws(norm_haystack)


def validate_citations(
    answer: str,
    source_chunks: list[dict] | None = None,
    source_documents: list[str] | None = None,
) -> CitationValidationResult:
    """Prüft, ob die in `answer` genannten Zitate in den Quellen auftauchen.

    Parameter:
        answer: Die zu prüfende Antwort (Klartext).
        source_chunks: Liste von Chunk-Dicts (mit `content` oder `text`).
        source_documents: Alternativ: Liste beliebiger Volltexte (Fallback).

    Rückgabe: `CitationValidationResult` mit Precision und Risiko-Level.

    Risiko-Schwellen:
        - precision ≥ 0.9 → low
        - precision ≥ 0.6 → medium
        - sonst          → high
    """
    cited = extract_citations(answer)
    if not cited:
        return CitationValidationResult(
            cited_references=[],
            verified_references=[],
            unverified_references=[],
            citation_precision=1.0,
            hallucination_risk="low",
        )

    # Quell-Texte zusammenstellen
    haystack_parts: list[str] = []
    if source_chunks:
        for c in source_chunks:
            haystack_parts.append(c.get("content") or c.get("text") or "")
    if source_documents:
        haystack_parts.extend(source_documents)
    combined = "\n\n".join(h for h in haystack_parts if h)

    verified: list[str] = []
    unverified: list[str] = []
    seen: set[str] = set()
    for c in cited:
        key = c.lower()
        if key in seen:
            continue
        seen.add(key)
        if combined and _is_citation_in_text(c, combined):
            verified.append(c)
        else:
            unverified.append(c)

    unique_total = len(verified) + len(unverified)
    precision = len(verified) / unique_total if unique_total else 1.0

    if precision >= 0.9:
        risk = "low"
    elif precision >= 0.6:
        risk = "medium"
    else:
        risk = "high"

    return CitationValidationResult(
        cited_references=cited,
        verified_references=verified,
        unverified_references=unverified,
        citation_precision=precision,
        hallucination_risk=risk,
    )
