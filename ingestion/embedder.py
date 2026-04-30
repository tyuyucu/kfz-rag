from openai import OpenAI
from config import OPENAI_API_KEY, EMBEDDING_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)


def embed_texts(texts: list[str]) -> list[list[float]]:
    """holt embeddings für mehrere texte über die openai api.

    läuft in paketen zu 100 einträgen (api-limit: 2048).
    """
    all_embeddings = []
    batch_size = 100

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


def embed_query(query: str) -> list[float]:
    """holt ein embedding für eine einzelne suchanfrage."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query
    )
    return response.data[0].embedding
