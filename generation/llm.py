"""llm-adapter buendelt chat-aufrufe

routing:
1) wenn in der sidebar ein api-key gesetzt wurde
   wird llm_client mit diesem key genutzt (OpenAI Groq Gemini)
2) sonst fallback auf lokales Ollama (https://ollama.com)

sonderfall evaluation: chat_with_usage laeuft fest ueber OpenAI
weil die auswertungen reproduzierbar bleiben sollen
und token-metriken gebraucht werden

embeddings laufen immer ueber OpenAI weil die chunks in der db
mit text-embedding-3-small vektorisiert sind
"""

from __future__ import annotations

import os
from collections.abc import Iterator

from openai import OpenAI

from config import OPENAI_API_KEY, LLM_MODEL

# default-client einmal initialisiert
# wird bei aufrufen ohne user-key genutzt (eval-pfad)
_default_openai = OpenAI(api_key=OPENAI_API_KEY)

DEFAULT_MAX_TOKENS = 2048

# default-modell fuer den ollama-fallback
# wird beim ersten aufruf automatisch nachgeladen (~2 gb)
OLLAMA_DEFAULT_MODEL = "llama3.2:3b"

# merker damit wir den pull nicht bei jeder anfrage neu pruefen
_ollama_model_checked: set[str] = set()


def _session_creds() -> tuple[str, str] | None:
    """liest user-key und provider aus streamlit-session-state"""
    try:
        import streamlit as st
        # ausserhalb eines streamlit-runs gibt es keinen session-state
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
    """erzeugt einen LLMClient mit dem user-key
    ohne os.environ dauerhaft zu modifizieren
    """
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
    """stellt sicher dass das ollama-modell lokal verfuegbar ist
    laedt es beim ersten aufruf automatisch nach
    wenn ollama nicht erreichbar ist wird die exception geschluckt
    der eigentliche chat-call wirft die meldung dann sauber selbst
    """
    if model in _ollama_model_checked:
        return
    try:
        import ollama
        # verfuegbare modelle abfragen
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
            pass  # wenn list scheitert versuchen wir trotzdem den pull

        # streamlit-spinner anzeigen falls verfuegbar
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
        # weder list noch pull moeglich
        # lass den eigentlichen chat-call den fehler werfen
        pass


def _get_chat_client(*, temperature: float, max_tokens: int):
    """liefert einen LLMClient fuer die aktuelle session
    mit user-key: gewaehlter provider
    ohne user-key: lokales ollama
    """
    from llm_client import LLMClient

    creds = _session_creds()
    if creds:
        api_key, provider = creds
        return _make_llm_client(api_key, provider,
                                temperature=temperature, max_tokens=max_tokens)

    # ohne user-key explizit ollama erzwingen
    # damit LLMClient nicht stillschweigend auf den env-key zurueckfaellt
    _ensure_ollama_model(OLLAMA_DEFAULT_MODEL)
    return LLMClient(
        api_choice="ollama",
        llm=OLLAMA_DEFAULT_MODEL,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def chat(messages: list[dict], *, temperature: float = 0.7,
         max_tokens: int = DEFAULT_MAX_TOKENS) -> str:
    """single-shot chat-completion
    nutzt user-key falls gesetzt sonst lokales ollama
    """
    client = _get_chat_client(temperature=temperature, max_tokens=max_tokens)
    return client.chat_completion(messages)


def chat_stream(messages: list[dict], *, temperature: float = 0.7,
                max_tokens: int = DEFAULT_MAX_TOKENS) -> Iterator[str]:
    """streamt eine chat-antwort token fuer token"""
    client = _get_chat_client(temperature=temperature, max_tokens=max_tokens)
    yield from client.chat_completion_stream(messages)


def chat_with_usage(messages: list[dict], *, temperature: float = 0.2
                    ) -> tuple[str, dict | None]:
    """wie chat aber gibt zusaetzlich die openai-usage-metriken zurueck
    nur im eval-pfad verwendet daher fest auf openai
    """
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
