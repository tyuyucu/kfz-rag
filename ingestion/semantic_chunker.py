"""Semantisches Chunking als alternative Ingestion-Strategie.

Nutzt `langchain_experimental.text_splitter.SemanticChunker`: statt
statischer Längen (RecursiveCharacterTextSplitter) werden Chunk-
Grenzen an Stellen gesetzt, an denen die Embeddings aufeinander-
folgender Sätze signifikant voneinander abweichen.

Erzeugt einheitliche `{content, page_number, chunk_index}`-Dicts wie
`ingestion.chunker.chunk_pages` — damit ist der restliche Ingestion-
Weg unverändert.

`langchain_experimental` ist eine optionale Abhängigkeit (siehe
`requirements.txt`): wenn das Paket nicht installiert ist, wirft der
Aufruf eine aussagekräftige Fehlermeldung.
"""

from __future__ import annotations

import logging

from config import OPENAI_API_KEY

logger = logging.getLogger(__name__)


def _build_semantic_splitter(*, breakpoint_threshold_type: str = "percentile"):
    try:
        from langchain_experimental.text_splitter import SemanticChunker
        from langchain_openai import OpenAIEmbeddings
    except ImportError as e:
        raise ImportError(
            "langchain-experimental oder langchain-openai fehlt. "
            "Installation: `pip install langchain-experimental`."
        ) from e

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY,
    )
    return SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type=breakpoint_threshold_type,
    )


def chunk_pages_semantic(
    pages: list[dict],
    *,
    breakpoint_threshold_type: str = "percentile",
) -> list[dict]:
    """Semantisches Chunking über seitenweise Eingabe.

    Jede Seite wird unabhängig semantisch geschnitten, die Page-Zuordnung
    bleibt also erhalten. Dies ist ein bewusster Kompromiss zwischen
    „global semantisch“ und Rückverfolgbarkeit für Quellenangaben.
    """
    splitter = _build_semantic_splitter(
        breakpoint_threshold_type=breakpoint_threshold_type
    )

    chunks: list[dict] = []
    chunk_index = 0
    for page in pages:
        text = page.get("text", "") or ""
        if not text.strip():
            continue
        try:
            parts = splitter.split_text(text)
        except Exception as e:
            logger.warning(
                "SemanticChunker schlug auf Seite %s fehl, überspringe: %s",
                page.get("page_number"),
                e,
            )
            continue
        for part in parts:
            part = part.strip()
            if not part:
                continue
            chunks.append({
                "content": part,
                "page_number": page.get("page_number"),
                "chunk_index": chunk_index,
            })
            chunk_index += 1
    return chunks
