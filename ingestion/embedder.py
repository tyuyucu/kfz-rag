"""embedding-layer
nutzt text-embedding-3-small ueber die openai-api
der schluessel wird im wizard hinterlegt und in app_config gespeichert
skripte ausserhalb der app fallen auf OPENAI_API_KEY env zurueck
"""

from __future__ import annotations

from config import EMBEDDING_DIMENSION, EMBEDDING_MODEL


# wird weiterhin exportiert fuer ui-anzeige
PROVIDER_OPENAI = "openai"


def _resolve_openai_key() -> str | None:
    """bevorzugt den im wizard hinterlegten schluessel
    faellt sonst auf die env-variable zurueck
    """
    try:
        from db.database import get_config
        key = get_config("openai_embedding_key")
        if key:
            return key
    except Exception:
        pass
    from config import OPENAI_API_KEY
    return OPENAI_API_KEY


# cache fuer den openai-client
# wird geleert wenn der schluessel wechselt
_openai_client = None
_openai_key_cache: str | None = None


def _get_openai_client():
    global _openai_client, _openai_key_cache
    key = _resolve_openai_key()
    if not key:
        raise RuntimeError(
            "kein OpenAI-Schluessel fuer Embeddings hinterlegt. "
            "Bitte im Setup-Wizard eintragen oder OPENAI_API_KEY setzen."
        )
    if _openai_client is None or _openai_key_cache != key:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=key)
        _openai_key_cache = key
    return _openai_client


def embed_texts(texts: list[str]) -> list[list[float]]:
    """embeddings fuer eine liste von texten
    verarbeitet in batches à 100 (api-limit liegt bei 2048)
    """
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
    """embedding fuer eine einzelne suchanfrage"""
    return embed_texts([query])[0]


def get_active_provider() -> str:
    """helper fuer ui aktuell immer openai"""
    return PROVIDER_OPENAI


def reset_provider_caches() -> None:
    """leert den internen client-cache
    wird nach key-wechsel oder reset aufgerufen
    """
    global _openai_client, _openai_key_cache
    _openai_client = None
    _openai_key_cache = None


def verify_openai_key(key: str) -> tuple[bool, str | None]:
    """mini-embed-aufruf zur key-validierung
    returns (ok fehlermeldung)
    """
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
