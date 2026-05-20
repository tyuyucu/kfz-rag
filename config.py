import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Streamlit Cloud: lade Secrets in os.environ, falls vorhanden.
# Lokal -> keine secrets.toml -> Aufruf ueberspringen (sonst Streamlit-Warnung +
# Konflikt mit set_page_config in app.py).
_secrets_paths = [
    Path.home() / ".streamlit" / "secrets.toml",
    Path(__file__).parent / ".streamlit" / "secrets.toml",
]
if any(p.exists() for p in _secrets_paths):
    try:
        import streamlit as st
        for _key, _val in st.secrets.items():
            if isinstance(_val, str) and _key not in os.environ:
                os.environ[_key] = _val
    except Exception:
        pass

# OpenAI
# Hinweis: OPENAI_API_KEY hier ist nur der Eval- bzw. Bestands-Fallback.
# Der Embedding-Schlüssel des Studierenden landet beim First-Run-Wizard in
# der `app_config`-Tabelle und wird vom Embedder bevorzugt benutzt.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# Modellname für den OpenAI-Embedding-Pfad. Der lokale Qwen2-Pfad nutzt
# eine eigene Konstante in `ingestion/embedder.py`.
EMBEDDING_MODEL = "text-embedding-3-small"
# Dimension der Vektoren in der DB-Spalte `chunks.embedding`. Beide
# unterstützten Embedding-Provider (OpenAI 3-small, GTE-Qwen2-1.5B)
# liefern 1536 Dimensionen — Schema bleibt für beide Pfade identisch.
EMBEDDING_DIMENSION = 1536
LLM_MODEL = "gpt-4o-mini"

# Database
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://raguser:ragpassword@localhost:5432/ragdb"
)

# Chunking
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Retrieval
MULTI_QUERY_COUNT = 3
HYBRID_SEARCH_TOP_K = 20
RRF_K = 60
RERANK_TOP_K = 5
RERANK_CANDIDATE_POOL = 20  # wie viele RRF-Top-Kandidaten an den Cross-Encoder gehen
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# Paths
DOCUMENTS_DIR = os.path.join(os.path.dirname(__file__), "documents")
