from generation.llm import chat


def rewrite_question_with_history(
    chat_history: list[dict],
    current_question: str,
) -> str:
    """Formuliert die aktuelle Frage unter Berücksichtigung der Chat-History um.

    Bei Folgefragen wie "Was bedeutet das genau?" wird der Kontext
    aus der History einbezogen, um eine eigenständige Suchanfrage zu erstellen.
    """
    if not chat_history:
        return current_question

    history_text = "\n".join(
        f"{msg['role'].capitalize()}: {msg['content']}"
        for msg in chat_history[-6:]
    )

    return chat(
        [
            {
                "role": "system",
                "content": (
                    "Du bist ein Assistent, der Folgefragen in eigenständige "
                    "Suchanfragen umformuliert. Nutze den Chatverlauf, um den "
                    "Kontext zu verstehen. Gib NUR die umformulierte Frage aus, "
                    "ohne Erklärungen. Wenn die Frage bereits eigenständig ist, "
                    "gib sie unverändert zurück."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Chatverlauf:\n{history_text}\n\n"
                    f"Aktuelle Frage: {current_question}\n\n"
                    f"Umformulierte eigenständige Frage:"
                ),
            },
        ],
        temperature=0.0,
        max_tokens=256,
    )
