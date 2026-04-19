import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP


def _is_toc_page(text: str) -> bool:
    """Erkennt Inhaltsverzeichnis-Seiten anhand typischer Muster.

    Typisch: viele Zeilen mit Punktreihen (....) oder Seitenzahlen am Ende.
    """
    lines = text.strip().split("\n")
    if len(lines) < 3:
        return False

    toc_pattern = re.compile(r'\.{4,}|\.\s*\d+\s*$')
    toc_lines = sum(1 for line in lines if toc_pattern.search(line))

    return toc_lines >= len(lines) * 0.4


def chunk_pages(pages: list[dict]) -> list[dict]:
    """Teilt seitenweisen Text in Chunks auf.

    Input: Liste von dicts mit keys: page_number, text
    Output: Liste von dicts mit keys: content, page_number, chunk_index

    Inhaltsverzeichnis-Seiten werden übersprungen.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    all_chunks = []
    chunk_index = 0

    for page in pages:
        if _is_toc_page(page["text"]):
            continue

        texts = splitter.split_text(page["text"])
        for text in texts:
            all_chunks.append({
                "content": text,
                "page_number": page["page_number"],
                "chunk_index": chunk_index,
            })
            chunk_index += 1

    return all_chunks
