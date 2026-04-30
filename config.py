import os
from dotenv import load_dotenv

load_dotenv()

# streamlit cloud: lade secrets in os.environ, falls vorhanden
try:
    import streamlit as st
    for _key, _val in st.secrets.items():
        if isinstance(_val, str) and _key not in os.environ:
            os.environ[_key] = _val
except Exception:
    pass

# openai einstellungen
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536
LLM_MODEL = "gpt-4o-mini"

# datenbank einstellungen
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://raguser:ragpassword@localhost:5432/ragdb"
)

# chunking einstellungen
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# retrieval einstellungen
MULTI_QUERY_COUNT = 3
HYBRID_SEARCH_TOP_K = 20
RRF_K = 60
RERANK_TOP_K = 5
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# pfade
DOCUMENTS_DIR = os.path.join(os.path.dirname(__file__), "documents")
