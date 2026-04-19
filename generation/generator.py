from openai import OpenAI
from config import OPENAI_API_KEY, LLM_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEM_PROMPT = """Du bist ein Lern-Assistent für die Vorlesung "Kfz-Haftpflichtversicherung" an einer deutschen Hochschule. Deine Aufgabe ist es, Studierenden beim Lernen zu helfen.

Regeln:
1. Antworte AUSSCHLIESSLICH auf Basis der bereitgestellten Kontext-Abschnitte.
2. Wenn die Frage unklar, unvollständig oder nicht zum Thema passt, bitte den Studierenden seine Frage zu präzisieren. Nenne in diesem Fall KEINE Quellen.
3. Wenn die Antwort nicht im Kontext enthalten ist, sage das ehrlich und rate nicht. Nenne auch hier KEINE Quellen.
4. Antworte immer auf Deutsch.
5. Nenne am Ende deiner Antwort die verwendeten Quellen (Dateiname und Seitenzahl) — aber NUR wenn du eine inhaltliche Antwort gibst.
6. Erkläre juristische Fachbegriffe verständlich für Studierende.
7. Strukturiere längere Antworten mit Aufzählungen oder Absätzen für bessere Lesbarkeit."""

GREETING_PROMPT = """Du bist ein freundlicher Lern-Assistent für die Vorlesung "Kfz-Haftpflichtversicherung" an einer deutschen Hochschule.

Der Studierende hat dich gerade begrüßt oder Smalltalk gemacht. Antworte freundlich und kurz auf Deutsch. Weise darauf hin, dass du bei Fragen zur Kfz-Haftpflichtversicherung helfen kannst. Halte die Antwort kurz (2-3 Sätze). Gib KEINE Quellen an."""

SPARRING_PROMPT = """Du bist ein sokratischer Lernassistent für die Vorlesung "Kfz-Haftpflichtversicherung". \
Deine Aufgabe ist es, den Studierenden durch gezieltes Fragen zum selbstständigen Denken \
zu führen – du gibst KEINE direkten Antworten.

REGELN:
- Beantworte Fragen NIEMALS direkt
- Stelle stattdessen 1-2 gezielte Gegenfragen die den Studierenden zur Antwort führen
- Wenn der Studierende eine richtige Teilantwort gibt: bestätige kurz und vertiefe mit einer Folgefrage
- Wenn der Studierende falsch liegt: widerspreche nicht direkt, sondern frage "Was steht dazu in §X?" oder "Wie würdest du das mit dem Fall Y vereinbaren?"
- Nutze ausschließlich die Inhalte aus der bereitgestellten Kontext-Abschnitte (Wissensbasis)
- Beende die sokratische Sequenz wenn der Studierende die Antwort selbst erarbeitet hat mit einem kurzen Lob und einer Zusammenfassung
- Bleibe immer im Kontext Kfz-Haftpflichtversicherung
- Antworte immer auf Deutsch
- Nenne KEINE Quellen in deiner Antwort — die Quellenangaben werden separat angezeigt"""


def is_greeting(query: str) -> bool:
    """Prüft ob eine Nachricht eine reine Begrüßung oder Smalltalk ist.

    Gibt nur True zurück wenn die Nachricht KURZ ist und nur Begrüßung enthält.
    Nachrichten mit Fachfragen nach der Begrüßung werden nicht als Greeting erkannt.
    """
    query_lower = query.strip().lower().rstrip("!?.,:; ")
    # Nur kurze Nachrichten (max 60 Zeichen) können reine Begrüßungen sein
    if len(query_lower) > 60:
        return False
    # Optionale Präfixe entfernen ("ja danke dir" → "danke dir")
    prefixes = ["ja ", "ok ", "okay ", "jo ", "jap ", "gut ", "super ", "klar "]
    cleaned = query_lower
    for prefix in prefixes:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
            break

    greetings = [
        "hi", "hallo", "hey", "moin", "servus", "guten tag",
        "guten morgen", "guten abend", "guten nachmittag",
        "na", "yo", "hello", "hej", "grüß gott", "grüezi",
        "was geht", "wie geht", "wie gehts", "wie geht's",
        "danke", "dankeschön", "vielen dank", "tschüss",
        "bye", "ciao", "tschau", "bis dann", "auf wiedersehen",
        "danke dir", "danke schön", "alles klar"
    ]
    return query_lower in greetings or cleaned in greetings


def generate_greeting_response(query: str, chat_history: list[dict] | None = None) -> str:
    """Generiert eine freundliche Antwort auf Begrüßungen ohne RAG."""
    messages = [{"role": "system", "content": GREETING_PROMPT}]

    if chat_history:
        for msg in chat_history[-4:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": query})

    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.7,
        messages=messages
    )

    return response.choices[0].message.content.strip()


def generate_sparring_response(
    query: str,
    context_chunks: list[dict],
    chat_history: list[dict] | None = None
) -> str:
    """Generiert eine sokratische Antwort — keine direkten Antworten, nur Gegenfragen."""
    context_text = "\n\n---\n\n".join(
        f"[Quelle: {c['filename']}, Seite {c['page_number']}]\n{c['content']}"
        for c in context_chunks
    )

    messages = [{"role": "system", "content": SPARRING_PROMPT}]

    if chat_history:
        for msg in chat_history[-6:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

    user_message = (
        f"Kontext:\n{context_text}\n\n"
        f"Nachricht des Studierenden: {query}"
    )
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.4,
        messages=messages
    )

    return response.choices[0].message.content.strip()


def generate_answer(
    query: str,
    context_chunks: list[dict],
    chat_history: list[dict] | None = None
) -> str:
    """Generiert eine Antwort basierend auf den Kontext-Chunks."""
    context_text = "\n\n---\n\n".join(
        f"[Quelle: {c['filename']}, Seite {c['page_number']}]\n{c['content']}"
        for c in context_chunks
    )

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if chat_history:
        for msg in chat_history[-6:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

    user_message = (
        f"Kontext:\n{context_text}\n\n"
        f"Frage: {query}"
    )
    messages.append({"role": "user", "content": user_message})

    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.2,
        messages=messages
    )

    return response.choices[0].message.content.strip()
