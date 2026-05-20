import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# secrets in os.environ laden falls vorhanden
# lokal gibt es keine secrets.toml also ueberspringen sonst kollidiert
# st.secrets.items mit set_page_config
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
# OPENAI_API_KEY hier ist nur fallback
# der eigentliche embedding-key kommt aus app_config (gesetzt im wizard)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
# vektordimension in der db-spalte chunks.embedding
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
RERANK_CANDIDATE_POOL = 20  # wie viele rrf-top-kandidaten an den cross-encoder gehen
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# Paths
DOCUMENTS_DIR = os.path.join(os.path.dirname(__file__), "documents")
