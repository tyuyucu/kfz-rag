from config import MULTI_QUERY_COUNT
from generation.llm import chat


def generate_query_variants(original_query: str) -> list[str]:
    """llm generiert mehrere varianten der suchanfrage
    verbessert recall durch unterschiedliche formulierungen
    """
    text = chat(
        [
            {
                "role": "system",
                "content": (
                    f"Du bist ein Experte für Kfz-Haftpflichtversicherung. "
                    f"Generiere genau {MULTI_QUERY_COUNT} verschiedene Varianten "
                    f"der folgenden Suchanfrage. Jede Variante soll die gleiche "
                    f"Information suchen, aber anders formuliert sein. "
                    f"Gib nur die Varianten aus, eine pro Zeile, ohne Nummerierung."
                ),
            },
            {"role": "user", "content": original_query},
        ],
        temperature=0.7,
        max_tokens=512,
    )
    variants = [v.strip() for v in text.split("\n") if v.strip()]
    return [original_query] + variants[:MULTI_QUERY_COUNT]
