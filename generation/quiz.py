import json
import random
from openai import OpenAI
from config import OPENAI_API_KEY, LLM_MODEL
from ingestion.embedder import embed_query
from db.vector_store import semantic_search

client = OpenAI(api_key=OPENAI_API_KEY)


def generate_quiz_question(topic: str | None = None) -> dict | None:
    """Generiert eine Multiple-Choice-Frage basierend auf der Wissensbasis.

    Returns: dict mit keys: question, options (list), correct_index, explanation, source
    """
    if topic:
        query = topic
    else:
        # Zufälliges Thema aus der Wissensbasis wählen
        sample_topics = [
            "Haftpflichtversicherung Deckungsumfang",
            "Pflichtversicherungsgesetz",
            "Schadenregulierung Kfz",
            "Regress Versicherung",
            "AKB Allgemeine Bedingungen",
            "Halterhaftung Kfz",
            "Direktanspruch Geschädigter",
            "Obliegenheiten Versicherungsnehmer",
        ]
        query = random.choice(sample_topics)

    query_embedding = embed_query(query)
    chunks = semantic_search(query_embedding, top_k=3)

    if not chunks:
        return None

    context_text = "\n\n".join(c["content"] for c in chunks)

    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.7,
        messages=[
            {
                "role": "system",
                "content": (
                    "Du bist ein Dozent für Kfz-Haftpflichtversicherung. "
                    "Erstelle eine anspruchsvolle Multiple-Choice-Frage basierend "
                    "auf dem gegebenen Kontext. Die Frage soll das Verständnis prüfen, "
                    "nicht nur Faktenwissen abfragen.\n\n"
                    "Antworte AUSSCHLIESSLICH im folgenden JSON-Format:\n"
                    '{"question": "Die Frage", '
                    '"options": ["Option A", "Option B", "Option C", "Option D"], '
                    '"correct_index": 0, '
                    '"explanation": "Erklärung warum die richtige Antwort korrekt ist"}'
                )
            },
            {
                "role": "user",
                "content": f"Kontext:\n{context_text}\n\nErstelle eine Multiple-Choice-Frage:"
            }
        ]
    )

    try:
        answer_text = response.choices[0].message.content.strip()
        if "```json" in answer_text:
            answer_text = answer_text.split("```json")[1].split("```")[0].strip()
        elif "```" in answer_text:
            answer_text = answer_text.split("```")[1].split("```")[0].strip()

        quiz_data = json.loads(answer_text)
        # Alle verwendeten Quellen auflisten (dedupliziert)
        sources = []
        seen = set()
        for c in chunks:
            key = (c["filename"], c["page_number"])
            if key not in seen:
                seen.add(key)
                sources.append(f"{c['filename']}, Seite {c['page_number']}")
        quiz_data["source"] = " | ".join(sources)
        return quiz_data
    except (json.JSONDecodeError, KeyError, IndexError):
        return None
