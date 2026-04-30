from openai import OpenAI
from config import OPENAI_API_KEY, LLM_MODEL

client = OpenAI(api_key=OPENAI_API_KEY)


def rewrite_question_with_history(
    chat_history: list[dict],
    current_question: str
) -> str:
    """formuliert die aktuelle frage mit chatverlauf neu.

    bei folgefragen wie "was bedeutet das genau?" wird der verlauf
    genutzt, damit eine eigenständige suchfrage entsteht.
    """
    if not chat_history:
        return current_question

    history_text = "\n".join(
        f"{msg['role'].capitalize()}: {msg['content']}"
        for msg in chat_history[-6:]
    )

    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Du bist ein Assistent, der Folgefragen in eigenständige "
                    "Suchanfragen umformuliert. Nutze den Chatverlauf, um den "
                    "Kontext zu verstehen. Gib NUR die umformulierte Frage aus, "
                    "ohne Erklärungen. Wenn die Frage bereits eigenständig ist, "
                    "gib sie unverändert zurück."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Chatverlauf:\n{history_text}\n\n"
                    f"Aktuelle Frage: {current_question}\n\n"
                    f"Umformulierte eigenständige Frage:"
                )
            }
        ]
    )

    return response.choices[0].message.content.strip()
