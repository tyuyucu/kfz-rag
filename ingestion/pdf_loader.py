import fitz  # PyMuPDF
import hashlib


def extract_text_from_pdf(filepath: str) -> list[dict]:
    """extrahiert text seitenweise aus einer pdf
    returns: liste von dicts mit page_number und text
    """
    doc = fitz.open(filepath)
    pages = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        if text.strip():
            pages.append({
                "page_number": page_num + 1,
                "text": text.strip()
            })

    doc.close()
    return pages


def get_page_count(filepath: str) -> int:
    doc = fitz.open(filepath)
    count = len(doc)
    doc.close()
    return count


def compute_file_hash(filepath: str) -> str:
    """sha-256 hash einer datei fuer aenderungserkennung"""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for block in iter(lambda: f.read(8192), b""):
            sha256.update(block)
    return sha256.hexdigest()
