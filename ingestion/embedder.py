"""Embedding-Layer.

Verwendet `text-embedding-3-small` über die OpenAI-API. Der Schlüssel
wird einmalig im First-Run-Wizard hinterlegt und in der `app_config`-
Tabelle persistiert; Eval-Skripte fallen auf die `OPENAI_API_KEY`-
Umgebungsvariable zurück.

Ein lokaler, kostenfreier Embedding-Pfad (z. B. GTE-Qwen2-1.5B oder
multilingual-e5-large-instruct) wurde evaluiert und im Rahmen der
Bachelorarbeit verworfen — Begründung siehe
`notes/design_decisions.md` („Embedding-Provider: OpenAI als
Default"). Kurzfassung: 1,5 Mrd-Parameter-Modelle benötigen auf
typischer Studi-CPU ~80–130 min für eine ~21 MB PDF und ~10 GB RAM,
was die Anwendung praktisch unbenutzbar macht. Kleinere Modelle
hätten ein abweichendes Vektorraum-Schema erfordert. Da die
OpenAI-Kosten für Studierende bei <0,01 € pro Setup liegen, ist der
API-Pfad der pragmatische Standard.
"""

from __future__ import annotations

from config import EMBEDDING_DIMENSION, EMBEDDING_MODEL


# Konstanten — werden aus Bestandsgründen weiterhin exportiert (Eval-
# Skripte, Tests). PROVIDER_OPENAI bleibt als Aktivanzeige für die UI;
# der frühere Mehr-Provider-Pfad ist entfernt.
PROVIDER_OPENAI = "openai"


def _resolve_openai_key() -> str | None:
    """Bevorzugt den im Wizard eingegebenen Embedding-Schlüssel aus der
    DB; fällt auf die Env-Variable zurück (Eval-Skripte, CI)."""
    try:
        from db.database import get_config
        key = get_config("openai_embedding_key")
        if key:
            return key
    except Exception:
        pass
    from config import OPENAI_API_KEY
    return OPENAI_API_KEY


# Cache für den OpenAI-Client. Wird geleert, wenn der Schlüssel wechselt.
_openai_client = None
_openai_key_cache: str | None = None


def _get_openai_client():
    global _openai_client, _openai_key_cache
    key = _resolve_openai_key()
    if not key:
        raise RuntimeError(
            "Kein OpenAI-API-Schlüssel für Embeddings hinterlegt. "
            "Bitte im Setup-Wizard eintragen oder OPENAI_API_KEY setzen."
        )
    if _openai_client is None or _openai_key_cache != key:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=key)
        _openai_key_cache = key
    return _openai_client


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Erstellt Embeddings für eine Liste von Texten via OpenAI API.
    Verarbeitet in Batches à 100 (API-Limit liegt bei 2048)."""
    if not texts:
        return []
    client = _get_openai_client()
    out: list[list[float]] = []
    for i in range(0, len(texts), 100):
        batch = list(texts[i:i + 100])
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        out.extend(item.embedding for item in resp.data)
    return out


def embed_query(query: str) -> list[float]:
    """Embedded eine einzelne Suchanfrage."""
    return embed_texts([query])[0]


def get_active_provider() -> str:
    """Hilfsfunktion für UI/Logging — aktuell ausschließlich `openai`."""
    return PROVIDER_OPENAI


def reset_provider_caches() -> None:
    """Leert den internen OpenAI-Client-Cache. Wird nach Schlüssel-Wechsel
    oder beim Reset der Wissensbasis aufgerufen."""
    global _openai_client, _openai_key_cache
    _openai_client = None
    _openai_key_cache = None


def verify_openai_key(key: str) -> tuple[bool, str | None]:
    """Macht einen Mini-Embed-Aufruf, um den Schlüssel zu validieren.
    Returns (ok, fehlermeldung)."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input="test")
        if not resp.data or len(resp.data[0].embedding) != EMBEDDING_DIMENSION:
            return False, "Antwort hatte unerwartetes Format."
        return True, None
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


__all__ = [
    "PROVIDER_OPENAI",
    "embed_query",
    "embed_texts",
    "get_active_provider",
    "reset_provider_caches",
    "verify_openai_key",
]
