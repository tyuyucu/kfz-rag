import os
from ingestion.pdf_loader import extract_text_from_pdf, compute_file_hash, get_page_count
from ingestion.chunker import chunk_pages
from ingestion.embedder import embed_texts
from db.database import (
    get_document_by_filename, insert_document, delete_document, get_all_documents
)
from db.vector_store import insert_chunks
from config import DOCUMENTS_DIR


def ingest_document(filepath: str, filename: str) -> dict:
    """verarbeitet ein einzelnes pdf-dokument von anfang bis ende.

    rückgabe: dict mit infos zum ergebnis.
    """
    file_hash = compute_file_hash(filepath)

    existing = get_document_by_filename(filename)
    if existing and existing["file_hash"] == file_hash:
        return {"status": "unchanged", "filename": filename}

    if existing:
        delete_document(existing["id"])

    pages = extract_text_from_pdf(filepath)
    page_count = get_page_count(filepath)

    if not pages:
        return {"status": "error", "filename": filename, "message": "Kein Text extrahiert"}

    chunks = chunk_pages(pages)

    texts = [c["content"] for c in chunks]
    embeddings = embed_texts(texts)

    for i, chunk in enumerate(chunks):
        chunk["embedding"] = embeddings[i]

    doc_id = insert_document(filename, file_hash, page_count)
    insert_chunks(doc_id, chunks)

    return {
        "status": "ingested",
        "filename": filename,
        "pages": page_count,
        "chunks": len(chunks)
    }


def ingest_all_documents() -> list[dict]:
    """verarbeitet alle pdfs im documents-ordner."""
    results = []

    if not os.path.exists(DOCUMENTS_DIR):
        os.makedirs(DOCUMENTS_DIR)
        return results

    for filename in os.listdir(DOCUMENTS_DIR):
        if filename.lower().endswith(".pdf"):
            filepath = os.path.join(DOCUMENTS_DIR, filename)
            try:
                result = ingest_document(filepath, filename)
                results.append(result)
            except Exception as e:
                results.append({
                    "status": "error",
                    "filename": filename,
                    "message": str(e)
                })

    return results


def remove_document(filename: str) -> bool:
    """löscht ein dokument aus datenbank und dateisystem."""
    doc = get_document_by_filename(filename)
    if doc:
        delete_document(doc["id"])

    filepath = os.path.join(DOCUMENTS_DIR, filename)
    if os.path.exists(filepath):
        os.remove(filepath)
        return True
    return False
