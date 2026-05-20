from generation.llm import chat, chat_stream, chat_with_usage

SYSTEM_PROMPT = """Du bist ein Lern-Assistent zum Thema "Kfz-Haftpflichtversicherung". Deine Aufgabe ist es, beim Lernen zu helfen.

Regeln:
1. Antworte AUSSCHLIESSLICH auf Basis der bereitgestellten Kontext-Abschnitte.
2. Wenn die Frage unklar, unvollständig oder nicht zum Thema passt, bitte um Präzisierung. Nenne in diesem Fall KEINE Quellen.
3. Wenn die Antwort nicht im Kontext enthalten ist, sage das ehrlich und rate nicht. Nenne auch hier KEINE Quellen.
4. Antworte immer auf Deutsch.
5. Nenne am Ende deiner Antwort die verwendeten Quellen (Dateiname und Seitenzahl) — aber NUR wenn du eine inhaltliche Antwort gibst.
6. Erkläre juristische Fachbegriffe verständlich.
7. Strukturiere längere Antworten mit Aufzählungen oder Absätzen für bessere Lesbarkeit."""

GREETING_PROMPT = """Du bist ein freundlicher Lern-Assistent zum Thema "Kfz-Haftpflichtversicherung".

Du wurdest gerade begrüßt oder mit Smalltalk angesprochen. Antworte freundlich und kurz auf Deutsch. Weise darauf hin, dass du bei Fragen zur Kfz-Haftpflichtversicherung helfen kannst. Halte die Antwort kurz (2-3 Sätze). Gib KEINE Quellen an."""

SPARRING_PROMPT = """Du bist ein sokratischer Lern-Assistent zum Thema "Kfz-Haftpflichtversicherung". \
Deine Aufgabe ist es, durch gezieltes Fragen zum selbstständigen Denken zu führen.

UNTERSCHEIDE DEN FRAGETYP:

A) DEFINITIONS- ODER VERSTÄNDNISFRAGEN ("Was ist X?", "Was bedeutet X?", "Erkläre X", "Definiere X", "Was sagt § X?")
   → Hier liegt eine Wissenslücke vor. Reine Gegenfragen frustrieren.
   - Gib zuerst eine knappe sachliche Definition oder Erklärung (1-2 Sätze, ausschließlich aus dem Kontext).
   - Schließe direkt mit EINER sokratischen Folgefrage an, die das Verständnis vertieft (z.B. "Welche Konsequenz hat das, wenn …?" oder "Wie passt das zu Fall X?").

B) ANWENDUNGS-, BEWERTUNGS- UND SUBSUMTIONSFRAGEN ("Wann greift…?", "Wie würdest du Fall X einschätzen?", "Welche Rolle spielt…?")
   → Hier ist sokratisches Hinführen sinnvoll.
   - Stelle 1-2 gezielte Gegenfragen, die zur Antwort führen.
   - Beantworte die Frage NICHT vorab.

ALLGEMEINE REGELN:
- Wenn eine richtige Teilantwort kommt: bestätige kurz und vertiefe mit einer Folgefrage.
- Wenn die Antwort falsch ist: widerspreche nicht direkt, sondern frage "Was steht dazu in §X?" oder "Wie würdest du das mit dem Fall Y vereinbaren?"
- Wenn die Antwort vollständig erarbeitet wurde: kurzes Lob plus Zusammenfassung.
- Nutze ausschließlich die Inhalte aus den Kontext-Abschnitten.
- Bleibe immer im Kontext Kfz-Haftpflichtversicherung.
- Antworte immer auf Deutsch.
- Nenne KEINE Quellen in deiner Antwort — die Quellenangaben werden separat angezeigt."""


def is_greeting(query: str) -> bool:
    """prueft ob eine nachricht eine reine begruessung oder smalltalk ist
    nur True wenn die nachricht kurz ist und nur begruessung enthaelt
    nachrichten mit fachfragen nach der begruessung sind kein greeting
    """
    query_lower = query.strip().lower().rstrip("!?.,:; ")
    if len(query_lower) > 60:
        return False
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
    """freundliche antwort auf begruessungen ohne rag"""
    messages = [{"role": "system", "content": GREETING_PROMPT}]
    if chat_history:
        for msg in chat_history[-4:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": query})
    return chat(messages, temperature=0.7, max_tokens=256)


def _build_chat_messages(query, context_chunks, chat_history):
    context_text = "\n\n---\n\n".join(
        f"[Quelle: {c['filename']}, Seite {c['page_number']}]\n{c['content']}"
        for c in context_chunks
    )
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if chat_history:
        for msg in chat_history[-6:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({
        "role": "user",
        "content": f"Kontext:\n{context_text}\n\nFrage: {query}",
    })
    return messages


def _build_sparring_messages(query, context_chunks, chat_history):
    context_text = "\n\n---\n\n".join(
        f"[Quelle: {c['filename']}, Seite {c['page_number']}]\n{c['content']}"
        for c in context_chunks
    )
    messages = [{"role": "system", "content": SPARRING_PROMPT}]
    if chat_history:
        for msg in chat_history[-6:]:
            messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({
        "role": "user",
        "content": f"Kontext:\n{context_text}\n\nNachricht: {query}",
    })
    return messages


def generate_answer(
    query: str,
    context_chunks: list[dict],
    chat_history: list[dict] | None = None,
) -> str:
    """single-shot chat-antwort ohne streaming
    fuer fallback und tests
    """
    return chat(
        _build_chat_messages(query, context_chunks, chat_history),
        temperature=0.2,
    )


def generate_sparring_response(
    query: str,
    context_chunks: list[dict],
    chat_history: list[dict] | None = None,
) -> str:
    """single-shot sokratische antwort ohne streaming"""
    return chat(
        _build_sparring_messages(query, context_chunks, chat_history),
        temperature=0.4,
    )


def generate_answer_stream(
    query: str,
    context_chunks: list[dict],
    chat_history: list[dict] | None = None,
    *,
    temperature: float = 0.2,
):
    """streamt die antwort token fuer token"""
    yield from chat_stream(
        _build_chat_messages(query, context_chunks, chat_history),
        temperature=temperature,
    )


def generate_sparring_stream(
    query: str,
    context_chunks: list[dict],
    chat_history: list[dict] | None = None,
    *,
    temperature: float = 0.4,
):
    """streamt die sokratische antwort token fuer token"""
    yield from chat_stream(
        _build_sparring_messages(query, context_chunks, chat_history),
        temperature=temperature,
    )


def generate_answer_with_usage(
    query: str,
    context_chunks: list[dict],
    chat_history: list[dict] | None = None,
    *,
    temperature: float = 0.2,
) -> tuple[str, dict | None]:
    """antwort plus usage-metriken (prompt_tokens completion_tokens total_tokens)
    fuer die evaluation
    """
    return chat_with_usage(
        _build_chat_messages(query, context_chunks, chat_history),
        temperature=temperature,
    )
