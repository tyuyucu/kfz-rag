import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHUNK_SIZE, CHUNK_OVERLAP


def _is_toc_page(text: str) -> bool:
    """erkennt seiten mit inhaltsverzeichnis über typische muster.

    typisch sind viele zeilen mit punktreihen oder seitenzahlen am ende.
    """
    lines = text.strip().split("\n")
    if len(lines) < 3:
        return False

    toc_pattern = re.compile(r'\.{4,}|\.\s*\d+\s*$')
    toc_lines = sum(1 for line in lines if toc_pattern.search(line))

    return toc_lines >= len(lines) * 0.4


def chunk_pages(pages: list[dict]) -> list[dict]:
    """teilt den text pro seite in kleinere abschnitte auf.

    eingabe: liste mit page_number und text.
    ausgabe: liste mit content, page_number und chunk_index.

    seiten mit inhaltsverzeichnis werden ausgelassen.
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
