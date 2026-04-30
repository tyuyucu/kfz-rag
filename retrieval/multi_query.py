from openai import OpenAI
from config import OPENAI_API_KEY, LLM_MODEL, MULTI_QUERY_COUNT

client = OpenAI(api_key=OPENAI_API_KEY)


def generate_query_variants(original_query: str) -> list[str]:
    """macht mehrere varianten einer suchanfrage mit dem llm.

    so werden durch verschiedene formulierungen oft mehr passende
    abschnitte gefunden.
    """
    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.7,
        messages=[
            {
                "role": "system",
                "content": (
                    f"Du bist ein Experte für Kfz-Haftpflichtversicherung. "
                    f"Generiere genau {MULTI_QUERY_COUNT} verschiedene Varianten "
                    f"der folgenden Suchanfrage. Jede Variante soll die gleiche "
                    f"Information suchen, aber anders formuliert sein. "
                    f"Gib nur die Varianten aus, eine pro Zeile, ohne Nummerierung."
                )
            },
            {"role": "user", "content": original_query}
        ]
    )

    variants_text = response.choices[0].message.content.strip()
    variants = [v.strip() for v in variants_text.split("\n") if v.strip()]

    # original-query immer mitnehmen
    all_queries = [original_query] + variants[:MULTI_QUERY_COUNT]
    return all_queries
