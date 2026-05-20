import json
import random
import re
from ingestion.embedder import embed_query
from db.vector_store import semantic_search
from generation.llm import chat


# pflichtfelder die das llm im json liefern muss
_REQUIRED_KEYS = ("question", "options", "correct_index", "explanation")


def _extract_json_object(text: str) -> str:
    """extrahiert das erste json-objekt aus der llm-antwort
    beruecksichtigt markdown-code-bloecke
    sonst greift es per regex auf den ersten {...}-block falls drumherum-text dabei ist
    """
    if "```json" in text:
        text = text.split("```json", 1)[1]
        if "```" in text:
            text = text.split("```", 1)[0]
    elif "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1]

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return match.group(0).strip()
    return text.strip()


def _validate_quiz_dict(data: dict) -> str | None:
    """strukturpruefung
    None bei ok sonst fehlermeldung
    """
    for key in _REQUIRED_KEYS:
        if key not in data:
            return f"Pflichtfeld '{key}' fehlt."
    if not isinstance(data["options"], list) or len(data["options"]) != 4:
        return "Es wurden nicht genau 4 Antwortoptionen geliefert."
    if not isinstance(data["correct_index"], int) or not (0 <= data["correct_index"] <= 3):
        return "correct_index muss 0..3 sein."
    if not all(isinstance(o, str) and o.strip() for o in data["options"]):
        return "Eine Antwortoption ist leer oder kein Text."
    return None


def generate_quiz_question(
    topic: str | None = None,
) -> tuple[dict | None, str | None]:
    """generiert eine multiple-choice-frage aus der wissensbasis

    returns (quiz_data error_message)
    erfolg: (dict None) mit keys question options correct_index explanation source source_chunks
    fehler: (None str) - meldung kann der ui direkt angezeigt werden
    """
    if topic:
        query = topic
    else:
        # zufaelliges thema aus der wissensbasis
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

    # retrieval
    try:
        query_embedding = embed_query(query)
        chunks = semantic_search(query_embedding, top_k=3)
    except Exception as e:
        return None, f"Suche in der Wissensbasis fehlgeschlagen: {e}"

    if not chunks:
        return None, (
            "Keine passenden Inhalte in der Wissensbasis gefunden. "
            "Bitte ein anderes Thema versuchen oder PDFs hochladen."
        )

    context_text = "\n\n".join(c["content"] for c in chunks)

    # llm-aufruf
    try:
        answer_text = chat(
            [
                {
                    "role": "system",
                    "content": (
                        "Du bist ein Dozent für Kfz-Haftpflichtversicherung. "
                        "Erstelle eine anspruchsvolle Multiple-Choice-Frage "
                        "basierend auf dem gegebenen Kontext. Die Frage soll "
                        "das Verständnis prüfen, nicht nur Faktenwissen.\n\n"
                        "Antworte AUSSCHLIESSLICH im folgenden JSON-Format, "
                        "ohne Drumherum-Text und ohne Markdown-Code-Blöcke:\n"
                        '{"question": "Die Frage", '
                        '"options": ["Option A", "Option B", "Option C", "Option D"], '
                        '"correct_index": 0, '
                        '"explanation": "Erklärung warum die richtige Antwort korrekt ist"}'
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Kontext:\n{context_text}\n\n"
                        "Erstelle eine Multiple-Choice-Frage als JSON:"
                    ),
                },
            ],
            temperature=0.7,
            max_tokens=1024,
        )
    except Exception as e:
        return None, f"Sprachmodell-Aufruf fehlgeschlagen: {e}"

    if not answer_text or not answer_text.strip():
        return None, "Sprachmodell hat eine leere Antwort geliefert."

    # json parsen
    raw_json = _extract_json_object(answer_text)
    try:
        quiz_data = json.loads(raw_json)
    except json.JSONDecodeError:
        return None, "Antwort konnte nicht als JSON gelesen werden."

    if not isinstance(quiz_data, dict):
        return None, "Antwort war kein JSON-Objekt."

    validation_error = _validate_quiz_dict(quiz_data)
    if validation_error:
        return None, f"Ungültige Frage-Struktur: {validation_error}"

    # quellen anhaengen
    sources_text: list[str] = []
    sources_struct: list[dict] = []
    seen: set[tuple[str, int]] = set()
    for c in chunks:
        key = (c["filename"], c["page_number"])
        if key not in seen:
            seen.add(key)
            sources_text.append(f"{c['filename']}, Seite {c['page_number']}")
            sources_struct.append({
                "filename": c["filename"],
                "page_number": c["page_number"],
            })
    quiz_data["source"] = " | ".join(sources_text)
    quiz_data["source_chunks"] = sources_struct
    return quiz_data, None
