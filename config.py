import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
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
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Paths
DOCUMENTS_DIR = os.path.join(os.path.dirname(__file__), "documents")
