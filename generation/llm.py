"""LLM-Adapter — bündelt Chat-Aufrufe.

Routing-Logik:
1. Hat der/die Studierende in der Sidebar einen eigenen API-Schlüssel +
   Provider hinterlegt (Streamlit-Session-State `user_api_key` und
   `user_provider`), wird `llm_client` mit diesem Schlüssel genutzt
   (OpenAI / Groq / Gemini).
2. Sonst: Fallback auf lokales Ollama. Dieses ist kostenlos, erfordert
   aber eine lokale Installation (https://ollama.com). So entstehen dem
   Anbieter keine Token-Kosten für fremde Anfragen.

Sonderfall Evaluation: `chat_with_usage` läuft fest über OpenAI, da die
Bachelorarbeits-Auswertungen reproduzierbar bleiben sollen und die
Token-Metriken benötigt werden.

Hinweis Embeddings: laufen zwingend über OpenAI, weil die Chunks in der
Datenbank mit `text-embedding-3-small` vektorisiert wurden. Embeddings
sind sehr günstig (~0,02 USD pro 1 Mio. Tokens) und werden vom Anbieter
getragen.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

from openai import OpenAI

from config import OPENAI_API_KEY, LLM_MODEL

# Default-Client: einmal initialisiert, wird bei Aufrufen ohne User-Key benutzt.
_default_openai = OpenAI(api_key=OPENAI_API_KEY)

DEFAULT_MAX_TOKENS = 2048

# Default-Modell für Ollama-Fallback. Wird beim ersten Aufruf automatisch
# nachgeladen, falls noch nicht lokal vorhanden (~2,0 GB).
# Begründung der 3B-Wahl siehe notes/design_decisions.md
# („Default-Modell für lokales Ollama").
OLLAMA_DEFAULT_MODEL = "llama3.2:3b"

# Merker, damit wir den Pull nicht bei jeder Anfrage erneut prüfen.
_ollama_model_checked: set[str] = set()


def _session_creds() -> tuple[str, str] | None:
    """Liest User-Key + Provider aus dem Streamlit-Session-State, falls vorhanden."""
    try:
        import streamlit as st
        # Außerhalb eines Streamlit-Skript-Runs (z.B. in Eval-Skripten) gibt es
        # keinen aktiven Session-State.
        if not st.runtime.exists():
            return None
        api_key = (st.session_state.get("user_api_key") or "").strip()
        provider = (st.session_state.get("user_provider") or "openai").strip().lower()
        if not api_key:
            return None
        return api_key, provider
    except Exception:
        return None


def _make_llm_client(api_key: str, provider: str, *, temperature: float,
                     max_tokens: int):
    """Erzeugt einen LLMClient mit dem User-Key, ohne os.environ dauerhaft zu modifizieren."""
    from llm_client import LLMClient

    env_var = f"{provider.upper()}_API_KEY"
    prev = os.environ.get(env_var)
    os.environ[env_var] = api_key
    try:
        return LLMClient(
            api_choice=provider,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
        )
    finally:
        if prev is None:
            os.environ.pop(env_var, None)
        else:
            os.environ[env_var] = prev


def _ensure_ollama_model(model: str = OLLAMA_DEFAULT_MODEL) -> None:
    """Sorgt dafür, dass das Ollama-Modell lokal verfügbar ist. Lädt es
    beim ersten Aufruf automatisch nach.

    Wenn Ollama nicht erreichbar ist oder der Pull scheitert, schluckt die
    Funktion die Exception — der eigentliche Chat-Aufruf wird die Fehler-
    meldung ohnehin sauber surface'n.
    """
    if model in _ollama_model_checked:
        return
    try:
        import ollama
        # Verfuegbare Modelle abfragen
        try:
            resp = ollama.list()
            existing = []
            for m in (resp.get("models", []) if isinstance(resp, dict)
                      else getattr(resp, "models", [])):
                name = (m.get("model") or m.get("name") or ""
                        if isinstance(m, dict) else getattr(m, "model", ""))
                existing.append(name)
            if any(n == model or n.startswith(f"{model}:") for n in existing):
                _ollama_model_checked.add(model)
                return
        except Exception:
            pass  # Wenn list() scheitert, gehen wir trotzdem auf pull

        # Streamlit-Status anzeigen, falls verfuegbar
        try:
            import streamlit as st
            if st.runtime.exists():
                with st.spinner(
                    f"Lade Ollama-Modell '{model}' herunter (~2,0 GB, "
                    f"einmalig)..."
                ):
                    ollama.pull(model)
            else:
                ollama.pull(model)
        except Exception:
            ollama.pull(model)
        _ollama_model_checked.add(model)
    except Exception:
        # Falls weder list noch pull moeglich war: lass den eigentlichen
        # Chat-Aufruf den Fehler werfen (klarer fuer den User).
        pass


def _get_chat_client(*, temperature: float, max_tokens: int):
    """Liefert einen LLMClient für die aktuelle Session.

    Mit User-Key: konfigurierter Provider (OpenAI/Groq/Gemini).
    Ohne User-Key: lokales Ollama (kostenlos, Installation erforderlich).
    Default-Ollama-Modell wird bei Bedarf automatisch nachgeladen.
    """
    from llm_client import LLMClient

    creds = _session_creds()
    if creds:
        api_key, provider = creds
        return _make_llm_client(api_key, provider,
                                temperature=temperature, max_tokens=max_tokens)

    # Kein User-Key: explizit Ollama erzwingen, damit LLMClient nicht still-
    # schweigend auf den Projekt-OpenAI-Key aus der env zurueckfaellt.
    _ensure_ollama_model(OLLAMA_DEFAULT_MODEL)
    return LLMClient(
        api_choice="ollama",
        llm=OLLAMA_DEFAULT_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def chat(messages: list[dict], *, temperature: float = 0.7,
         max_tokens: int = DEFAULT_MAX_TOKENS) -> str:
    """Single-shot Chat-Completion. Nutzt User-Key falls gesetzt, sonst
    lokales Ollama."""
    client = _get_chat_client(temperature=temperature, max_tokens=max_tokens)
    return client.chat_completion(messages)


def chat_stream(messages: list[dict], *, temperature: float = 0.7,
                max_tokens: int = DEFAULT_MAX_TOKENS) -> Iterator[str]:
    """Streamt eine Chat-Antwort Token für Token."""
    client = _get_chat_client(temperature=temperature, max_tokens=max_tokens)
    yield from client.chat_completion_stream(messages)


def chat_with_usage(messages: list[dict], *, temperature: float = 0.2
                    ) -> tuple[str, dict | None]:
    """Wie `chat`, gibt aber zusätzlich die OpenAI-`usage`-Metriken zurück
    (prompt_tokens, completion_tokens). Wird ausschließlich im
    Evaluations-Pfad verwendet — daher fest auf OpenAI."""
    response = _default_openai.chat.completions.create(
        model=LLM_MODEL,
        temperature=temperature,
        messages=messages,
    )
    answer = response.choices[0].message.content.strip()
    usage = None
    if getattr(response, "usage", None) is not None:
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
    return answer, usage
