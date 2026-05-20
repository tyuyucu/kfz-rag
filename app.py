import os
import html
import json
import time
import threading
import uuid
from pathlib import Path
import streamlit as st

try:
    # private streamlit-api die einen background-thread an den session-ctx koppelt
    # damit zugriff auf st.session_state aus dem thread heraus funktioniert
    from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx
except Exception:
    add_script_run_ctx = None
    get_script_run_ctx = None

# secrets in os.environ laden bevor andere module geladen werden
# lokal gibt es keine secrets.toml also ueberspringen
_secrets_paths = [
    Path.home() / ".streamlit" / "secrets.toml",
    Path(__file__).parent / ".streamlit" / "secrets.toml",
]
if any(p.exists() for p in _secrets_paths):
    try:
        for _key, _val in st.secrets.items():
            if isinstance(_val, str) and _key not in os.environ:
                os.environ[_key] = _val
    except Exception:
        pass

from db.database import (
    init_db,
    get_all_documents,
    create_chat_session,
    save_chat_message,
    get_config,
    set_config,
    reset_knowledge_base,
)
from db.vector_store import get_chunk_count
from ingestion.pipeline import ingest_document, remove_document
from ingestion.embedder import (
    PROVIDER_OPENAI,
    get_active_provider,
    reset_provider_caches,
    verify_openai_key,
)
from retrieval.pipeline import retrieve, get_last_timings
from generation.history import rewrite_question_with_history
from generation.generator import (
    generate_answer,
    is_greeting,
    generate_greeting_response,
    generate_sparring_response,
    generate_answer_stream,
    generate_sparring_stream,
)
from generation.quiz import generate_quiz_question
from streamlit_pdf_viewer import pdf_viewer
from streamlit_local_storage import LocalStorage
from config import DOCUMENTS_DIR


# config-keys fuer den setup-wizard (in app_config-tabelle)
SETUP_FLAG_KEY = "setup_completed"
OPENAI_EMBED_KEY = "openai_embedding_key"


# module-level statt session-state weil background-thread session-state
# nicht zuverlaessig schreiben kann
# key ist session_id pro user
_prefetch_results: dict[int, tuple] = {}
_prefetch_lock = threading.Lock()


def _prefetch_next_quiz(session_id: int, topic: str | None) -> None:
    """startet einen background-thread der die naechste quiz-frage generiert
    ergebnis landet in _prefetch_results unter der session_id
    """
    if add_script_run_ctx is None or get_script_run_ctx is None:
        return
    ctx = get_script_run_ctx()
    if ctx is None:
        return

    def worker():
        try:
            add_script_run_ctx(threading.current_thread(), ctx)
        except Exception:
            pass
        try:
            data, error = generate_quiz_question(topic)
            with _prefetch_lock:
                _prefetch_results[session_id] = (data, error, topic)
        except Exception:
            pass

    t = threading.Thread(target=worker, daemon=True)
    t.start()


def _warmup_reranker_async() -> None:
    """laed den reranker beim app-start im hintergrund
    damit der erste retrieve-call nicht den cold-start hat (~5-10s)
    """
    if st.session_state.get("_reranker_warmup_started"):
        return
    st.session_state._reranker_warmup_started = True
    if add_script_run_ctx is None or get_script_run_ctx is None:
        return
    ctx = get_script_run_ctx()
    if ctx is None:
        return

    def worker():
        try:
            add_script_run_ctx(threading.current_thread(), ctx)
        except Exception:
            pass
        try:
            from retrieval.reranker import get_reranker
            get_reranker()  # triggert das lazy-load
        except Exception:
            pass

    threading.Thread(target=worker, daemon=True).start()


def _pop_prefetched_quiz(session_id: int, topic: str | None, *, wait_ms: int = 0):
    """liefert das prefetched ergebnis falls fuer session und topic vorhanden
    wartet bis zu wait_ms millisekunden falls der background-thread noch laeuft
    """
    deadline = time.time() + wait_ms / 1000
    while True:
        with _prefetch_lock:
            entry = _prefetch_results.pop(session_id, None)
        if entry is not None:
            data, error, prefetch_topic = entry
            if prefetch_topic != topic:
                return None
            return data, error
        if time.time() >= deadline:
            return None
        time.sleep(0.05)


# ── chats clientseitig im browser-localstorage halten ──
# pro mode (Chat/Sparring) eine liste von vergangenen chats
# jeder chat hat id title messages
# die letzten 20 werden behalten

_LS_CHATS_KEY = "kfz_chats"
_LS_MAX_CHATS = 20


def _ls() -> LocalStorage:
    """liefert die LocalStorage-instanz aus dem cache"""
    if "_ls_instance" not in st.session_state:
        st.session_state._ls_instance = LocalStorage()
    return st.session_state._ls_instance


def _chats_storage_key(mode: str | None) -> str:
    """key pro mode"""
    return f"{_LS_CHATS_KEY}_{mode or 'none'}"


def _new_chat_id() -> str:
    """eindeutige id pro chat im format YYYYMMDD-HHMMSS-<shortuuid>"""
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"


def _load_chats(mode: str | None) -> list[dict]:
    """alle gespeicherten chats fuer den mode
    leere liste wenn nichts gespeichert oder beim ersten render
    """
    if not mode or mode == "Quiz":
        return []
    try:
        raw = _ls().getItem(_chats_storage_key(mode))
        if not raw:
            return []
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def _save_chat(mode: str | None, chat_id: str, messages: list[dict]) -> None:
    """fuegt einen chat in die liste ein oder updated ihn
    titel = erste user-message oder fallback
    der zuletzt geschriebene chat steht oben
    """
    if not mode or mode == "Quiz" or not messages:
        return
    try:
        chats = _load_chats(mode)
        chats = [c for c in chats if c.get("id") != chat_id]
        title = "Chat"
        for m in messages:
            if m.get("role") == "user":
                t = (m.get("content") or "").strip()
                if t:
                    title = t[:60]
                    break
        chats.insert(0, {
            "id": chat_id,
            "title": title,
            "messages": messages,
        })
        chats = chats[:_LS_MAX_CHATS]
        _ls().setItem(_chats_storage_key(mode), json.dumps(chats))
    except Exception:
        pass


def _delete_chat(mode: str | None, chat_id: str) -> None:
    """entfernt einen einzelnen chat aus der liste"""
    if not mode or mode == "Quiz":
        return
    try:
        chats = _load_chats(mode)
        chats = [c for c in chats if c.get("id") != chat_id]
        _ls().setItem(_chats_storage_key(mode), json.dumps(chats))
    except Exception:
        pass


def _clear_chats(mode: str | None) -> None:
    """loescht alle chats fuer den mode"""
    if not mode or mode == "Quiz":
        return
    try:
        _ls().deleteItem(_chats_storage_key(mode))
    except Exception:
        pass


@st.dialog("📖 Quelle im PDF", width="large")
def _show_pdf_dialog(filepath: str, page: int, filename: str):
    """zeigt das pdf im dialog und springt direkt auf die zitierte seite"""
    st.caption(f"{filename} — Seite {page}")
    pdf_viewer(
        input=filepath,
        height=750,
        scroll_to_page=page,
        render_text=True,
    )


def _render_chunk_sources(context_chunks: list[dict], *, key_prefix: str) -> None:
    """rendert die quellen-liste mit vollem chunk-text und pdf-sprung-button"""
    for i, chunk in enumerate(context_chunks, 1):
        safe_filename = html.escape(chunk["filename"])
        safe_content = html.escape(chunk["content"])
        st.markdown(f"""
        <div class="source-chunk">
            <div class="source-header">
                📄 {safe_filename} — Seite {chunk["page_number"]}
                &nbsp;&middot;&nbsp; Relevanz: {chunk.get("rerank_score", 0):.3f}
            </div>
            <div class="source-text">{safe_content}</div>
        </div>
        """, unsafe_allow_html=True)

        pdf_path = os.path.join(DOCUMENTS_DIR, chunk["filename"])
        if os.path.exists(pdf_path):
            btn_key = f"pdf_{key_prefix}_{i}_{chunk['page_number']}"
            if st.button(
                f"📖 Seite {chunk['page_number']} im PDF öffnen",
                key=btn_key,
                use_container_width=True,
            ):
                _show_pdf_dialog(pdf_path, chunk["page_number"], chunk["filename"])


# ── first-run-setup-wizard ──

def _setup_completed() -> bool:
    """True wenn der wizard schon durchlaufen wurde
    db-getrieben damit die wahl auch nach browser-reload bleibt
    """
    try:
        return get_config(SETUP_FLAG_KEY) == "true"
    except Exception:
        # fallback: wenn die db nicht steht keinen wizard zeigen
        # der db-fehler wird in der sidebar gemeldet
        return True


def _wizard_step1_embedding(chunk_count_now: int) -> None:
    """schritt 1: openai-schluessel fuer embeddings hinterlegen"""
    st.markdown("""
    <div class="main-header">
        <span class="header-icon">⚙️</span>
        <h1>Erstkonfiguration</h1>
        <div class="header-sub">
            Hinterlege einmalig deinen OpenAI-Schlüssel — er wird
            ausschließlich lokal gespeichert und für die Vektor-
            Erzeugung deiner Lernmaterialien verwendet.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### OpenAI-API-Schlüssel")
    st.caption(
        "Die Suche arbeitet mit Embeddings — Vektor-Darstellungen deiner "
        "PDFs, erzeugt durch das Modell `text-embedding-3-small`. Kosten: "
        "ca. **0,01 €** einmalig für das Ingesten der PDFs, "
        "danach nur Bruchteile davon pro Suchanfrage. Den Schlüssel "
        "erstellst du auf "
        "[platform.openai.com/api-keys](https://platform.openai.com/api-keys)."
    )

    api_key_value = st.text_input(
        "Schlüssel",
        type="password",
        placeholder="sk-...",
        key="wizard_openai_embed_key",
        label_visibility="collapsed",
    )

    st.caption(
        "Wird verschlüsselt-im-Volume in deiner lokalen Postgres-DB "
        "abgelegt — verlässt deinen Rechner nie."
    )

    st.divider()

    if st.button("Weiter — Schlüssel speichern", type="primary",
                 use_container_width=True, key="wizard_step1_continue"):
        stripped = (api_key_value or "").strip()
        if not stripped:
            st.warning("Bitte den OpenAI-Schlüssel eingeben.")
            return
        with st.spinner("Schlüssel wird verifiziert…"):
            ok, err = verify_openai_key(stripped)
        if not ok:
            st.error(
                f"Schlüssel konnte nicht bestätigt werden: {err}. "
                "Bitte prüfen und erneut versuchen."
            )
            return
        set_config(OPENAI_EMBED_KEY, stripped)
        reset_provider_caches()
        st.session_state.wizard_step = "upload"
        st.rerun()


def _wizard_step2_upload() -> None:
    """schritt 2: pdfs hochladen und ingesten
    kann uebersprungen werden und spaeter ueber die sidebar nachgeholt werden
    """
    st.markdown("""
    <div class="main-header">
        <span class="header-icon">📚</span>
        <h1>Lernmaterialien hinzufügen</h1>
        <div class="header-sub">
            Lade jetzt die PDFs hoch, mit denen du arbeiten möchtest —
            oder überspringe diesen Schritt und füge sie später über die
            Sidebar hinzu.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    files = st.file_uploader(
        "PDFs hochladen",
        type=["pdf"],
        accept_multiple_files=True,
        key="wizard_pdf_uploader",
    )

    cols = st.columns([1, 1])
    with cols[0]:
        skip = st.button(
            "Überspringen — später hinzufügen",
            use_container_width=True,
            key="wizard_skip_upload",
        )
    with cols[1]:
        process = st.button(
            "PDFs verarbeiten",
            type="primary",
            use_container_width=True,
            key="wizard_process_upload",
            disabled=not files,
        )

    if skip:
        set_config(SETUP_FLAG_KEY, "true")
        st.session_state.wizard_step = "done"
        st.rerun()
        return

    if process and files:
        os.makedirs(DOCUMENTS_DIR, exist_ok=True)
        results: list[dict] = []
        progress = st.progress(0.0, text="Starte Ingest…")
        for i, uploaded in enumerate(files, 1):
            filepath = os.path.join(DOCUMENTS_DIR, uploaded.name)
            with open(filepath, "wb") as fh:
                fh.write(uploaded.getbuffer())
            progress.progress(
                (i - 0.5) / len(files),
                text=f"Verarbeite {uploaded.name} ({i}/{len(files)})…",
            )
            results.append(ingest_document(filepath, uploaded.name))
            progress.progress(i / len(files), text=f"{i}/{len(files)} fertig")

        # zusammenfassung
        ok = [r for r in results if r["status"] == "ingested"]
        unchanged = [r for r in results if r["status"] == "unchanged"]
        failed = [r for r in results if r["status"] not in ("ingested", "unchanged")]

        if ok:
            st.success(
                f"{len(ok)} PDF(s) verarbeitet — "
                f"{sum(r.get('chunks', 0) for r in ok)} Chunks im Index."
            )
        if unchanged:
            st.info(f"{len(unchanged)} PDF(s) bereits vorhanden, übersprungen.")
        if failed:
            st.error(
                "Fehler bei: "
                + ", ".join(f"{r.get('filename', '?')}" for r in failed)
            )

        set_config(SETUP_FLAG_KEY, "true")
        st.session_state.wizard_step = "done"
        # nicht sofort rerun damit die erfolgsmeldung sichtbar bleibt
        if st.button("App starten", type="primary",
                     use_container_width=True, key="wizard_finish"):
            st.rerun()


def _render_setup_wizard() -> None:
    """komplette wizard-ui
    verzweigt anhand wizard_step in der session
    """
    chunk_count_now = get_chunk_count()
    step = st.session_state.get("wizard_step")
    if step is None:
        # wenn die db schon chunks hat direkt zu upload sonst embedding
        step = "upload" if chunk_count_now > 0 else "embedding"
        st.session_state.wizard_step = step

    if step == "embedding":
        _wizard_step1_embedding(chunk_count_now)
    elif step == "upload":
        _wizard_step2_upload()
    else:
        # done - setup-flag setzen falls noch nicht passiert
        set_config(SETUP_FLAG_KEY, "true")
        st.rerun()


# ── konstanten ──
NO_ANSWER_HINTS = [
    "keine relevanten informationen",
    "nicht im kontext enthalten",
    "kann ich leider nicht beantworten",
    "keine ausreichenden informationen",
    "nicht genügend informationen",
    "unvollständig",
    "unklar formuliert",
    "bitte formuliere",
    "bitte stelle deine frage",
    "bitte präzisiere",
    "nicht eindeutig",
    "was genau möchtest du",
    "was möchtest du"
]

# ── seiten-config ──
st.set_page_config(
    page_title="Kfz-Haftpflicht Lern-Assistent",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ── custom css ──
st.markdown("""
<style>
    /* ── Allgemein ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* ── Dark-Mode universell erzwingen (Streamlit ignoriert Browser-Preference) ── */
    :root, [data-theme="light"], [data-theme="dark"] {
        --background-color: #0e1117 !important;
        --secondary-background-color: #1a1f2e !important;
        --text-color: #e2e8f0 !important;
        --primary-color: #6366f1 !important;
        color-scheme: dark !important;
    }
    html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #0e1117 !important;
        color: #e2e8f0 !important;
    }
    [data-testid="stHeader"] {
        background: rgba(14, 17, 23, 0.6) !important;
    }
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span,
    label, .stRadio label, .stCheckbox label,
    [data-testid="stWidgetLabel"], [data-testid="stMarkdownContainer"] {
        color: #e2e8f0 !important;
    }
    input, textarea, select,
    .stTextInput input, .stTextArea textarea, .stSelectbox div[role="combobox"],
    .stChatInput textarea {
        background-color: #1a1f2e !important;
        color: #e2e8f0 !important;
        border-color: rgba(255,255,255,0.1) !important;
    }
    input::placeholder, textarea::placeholder {
        color: #64748b !important;
    }
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploaderDropzone"] {
        background-color: #1a1f2e !important;
        border-color: rgba(255,255,255,0.1) !important;
    }
    [data-testid="stFileUploader"] section *,
    [data-testid="stFileUploaderDropzone"] * {
        color: #e2e8f0 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stColumn"] button {
        background: rgba(255,255,255,0.04) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        color: #94a3b8 !important;
    }
    [data-baseweb="radio"] > div:first-child,
    .stRadio [role="radio"] {
        background-color: #1a1f2e !important;
        border-color: rgba(255,255,255,0.25) !important;
    }
    [data-baseweb="radio"] input:checked + div,
    [data-baseweb="radio"][aria-checked="true"] > div:first-child {
        background-color: #1a1f2e !important;
        border-color: #ef4444 !important;
    }
    [data-testid="stBottomBlockContainer"],
    [data-testid="stBottom"],
    .stBottom,
    div[class*="stBottom"] {
        background-color: #0e1117 !important;
        background: #0e1117 !important;
        border-top: 1px solid rgba(255,255,255,0.05) !important;
    }
    [data-testid="stBottomBlockContainer"] > div,
    [data-testid="stBottom"] > div {
        background-color: transparent !important;
    }
    .stButton > button {
        background: #1a1f2e !important;
        color: #e2e8f0 !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
    }
    .stButton > button:hover {
        background: #232836 !important;
        border-color: rgba(99, 102, 241, 0.4) !important;
        color: #ffffff !important;
    }
    [data-testid="stForm"] button,
    button[data-testid="stFormSubmitButton"],
    button[data-testid^="stBaseButton-"][data-testid*="FormSubmit"] {
        background: rgba(99, 102, 241, 0.12) !important;
        border: 1px solid rgba(99, 102, 241, 0.35) !important;
        color: #e2e8f0 !important;
    }
    [data-testid="stForm"] button:hover,
    button[data-testid="stFormSubmitButton"]:hover,
    button[data-testid^="stBaseButton-"][data-testid*="FormSubmit"]:hover {
        background: rgba(99, 102, 241, 0.2) !important;
        border-color: rgba(99, 102, 241, 0.55) !important;
        color: #ffffff !important;
    }
    [data-testid="stTextInputRootElement"] button,
    .stTextInput button[kind="headerNoPadding"] {
        background: transparent !important;
        color: #94a3b8 !important;
        border: none !important;
    }
    [data-testid="stTextInputRootElement"] button:hover,
    .stTextInput button[kind="headerNoPadding"]:hover {
        color: #e2e8f0 !important;
        background: rgba(255,255,255,0.05) !important;
    }

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f2e 0%, #0f1320 100%);
        border-right: 1px solid rgba(99, 102, 241, 0.15);
    }

    section[data-testid="stSidebar"] .stMarkdown h1 {
        font-size: 1.3rem;
        letter-spacing: -0.02em;
    }

    /* ── Sidebar Branding-Button ── */
    /* ── Branding-Home-Button (erstes Element in Sidebar) ── */
    [data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:nth-child(1) button {
        background: transparent !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 14px !important;
        padding: 0.6rem 0.5rem !important;
        box-shadow: none !important;
        transition: all 0.2s !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: #e2e8f0 !important;
        letter-spacing: 0.02em !important;
    }
    [data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:nth-child(1) button:hover {
        background: rgba(99, 102, 241, 0.08) !important;
        border-color: rgba(99, 102, 241, 0.25) !important;
        color: #e2e8f0 !important;
    }
    [data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:nth-child(1) button p {
        white-space: pre-line !important;
        line-height: 1.4 !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
    }
    [data-testid="stSidebarUserContent"] [data-testid="stVerticalBlock"] > [data-testid="stElementContainer"]:nth-child(1) button p::first-line {
        font-size: 2.5rem !important;
        line-height: 2.2 !important;
    }

    /* ── Modus-Radio-Buttons gleichmäßig verteilen ── */
    section[data-testid="stSidebar"] [data-testid="stRadio"] > div[role="radiogroup"],
    section[data-testid="stSidebar"] [data-testid="stRadio"] > div {
        justify-content: space-evenly !important;
    }

    /* ── X-Button Hover (Löschen-Akzent) ── */
    section[data-testid="stSidebar"] [data-testid="stColumn"] button:hover {
        color: #f87171 !important;
        border-color: rgba(239, 68, 68, 0.5) !important;
        background: rgba(239, 68, 68, 0.1) !important;
    }

    /* ── Modus-Tabs ── */
    .mode-tabs {
        display: flex;
        gap: 0.5rem;
        margin: 0.8rem 0;
    }
    .mode-tab {
        flex: 1;
        text-align: center;
        padding: 0.55rem 0.8rem;
        border-radius: 10px;
        font-size: 0.85rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.2s;
        border: 1px solid transparent;
    }
    .mode-tab-active {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        box-shadow: 0 2px 12px rgba(99, 102, 241, 0.35);
    }
    .mode-tab-inactive {
        background: rgba(255,255,255,0.04);
        color: #94a3b8;
        border: 1px solid rgba(255,255,255,0.08);
    }

    /* ── Stat-Karten ── */
    .stat-row {
        display: flex;
        gap: 0.6rem;
        margin: 0.6rem 0;
    }
    .stat-card {
        flex: 1;
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 10px;
        padding: 0.7rem 0.8rem;
        text-align: center;
    }
    .stat-card .stat-value {
        font-size: 1.4rem;
        font-weight: 700;
        color: #e2e8f0;
    }
    .stat-card .stat-label {
        font-size: 0.7rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.15rem;
    }

    /* ── Dokument-Liste ── */
    .doc-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.55rem 0.7rem;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 8px;
        margin-bottom: 0.4rem;
        transition: background 0.15s;
    }
    .doc-item:hover {
        background: rgba(255,255,255,0.06);
    }
    .doc-info {
        flex: 1;
        min-width: 0;
    }
    .doc-name {
        font-size: 0.8rem;
        font-weight: 500;
        color: #cbd5e1;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .doc-meta {
        font-size: 0.68rem;
        color: #64748b;
        margin-top: 0.1rem;
    }

    /* ── Sidebar Section Headers ── */
    .sidebar-section {
        font-size: 0.72rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin: 1rem 0 0.5rem 0;
        padding-bottom: 0.3rem;
        border-bottom: 1px solid rgba(255,255,255,0.06);
    }

    /* ── Hauptbereich Header ── */
    .main-header {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
    }
    .main-header .header-icon {
        font-size: 3rem;
        margin-bottom: 0.5rem;
        display: block;
    }
    .main-header h1 {
        font-size: 1.6rem;
        font-weight: 700;
        color: #f1f5f9;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .main-header .header-sub {
        font-size: 0.9rem;
        color: #94a3b8;
        margin-top: 0.3rem;
    }

    /* ── Welcome Cards ── */
    .welcome-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 1rem;
        margin: 2rem 0;
    }
    .welcome-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 14px;
        padding: 1.4rem;
        text-align: center;
        transition: all 0.2s;
    }
    .welcome-card:hover {
        background: rgba(99, 102, 241, 0.06);
        border-color: rgba(99, 102, 241, 0.2);
        transform: translateY(-2px);
    }
    .welcome-card .card-icon {
        font-size: 2rem;
        margin-bottom: 0.6rem;
        display: block;
    }
    .welcome-card .card-title {
        font-size: 0.95rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 0.3rem;
    }
    .welcome-card .card-desc {
        font-size: 0.8rem;
        color: #94a3b8;
        line-height: 1.4;
    }

    /* ── Chat Messages ── */
    .stChatMessage {
        border-radius: 12px !important;
        border: 1px solid rgba(255,255,255,0.05) !important;
        margin-bottom: 0.8rem !important;
    }

    /* ── Chat Input ── */
    .stChatInput {
        border-radius: 14px !important;
    }
    .stChatInput > div {
        border-radius: 14px !important;
        border: 1px solid rgba(99, 102, 241, 0.3) !important;
        background: rgba(255,255,255,0.03) !important;
    }
    .stChatInput > div:focus-within {
        border-color: rgba(99, 102, 241, 0.6) !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.12) !important;
    }

    /* ── Quiz-Karte ── */
    .quiz-question-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 2rem;
        margin: 1rem 0;
    }
    .quiz-question-card .question-text {
        font-size: 1.15rem;
        font-weight: 600;
        color: #f1f5f9;
        line-height: 1.5;
        margin-bottom: 0.4rem;
    }
    .quiz-question-card .question-source {
        font-size: 0.75rem;
        color: #64748b;
    }

    /* ── Quiz Score Badge ── */
    .score-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(139, 92, 246, 0.1));
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 20px;
        padding: 0.4rem 1rem;
        font-size: 0.85rem;
        font-weight: 600;
        color: #a5b4fc;
        margin-bottom: 1rem;
    }

    /* ── Quiz Antwortoptionen nach Beantwortung ── */
    .quiz-option-answered {
        padding: 0.7rem 1rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .quiz-option-answered .option-letter {
        font-weight: 700;
        min-width: 1.5rem;
    }
    .quiz-option-correct {
        background: rgba(34, 197, 94, 0.12);
        border: 1px solid rgba(34, 197, 94, 0.4);
        color: #4ade80;
    }
    .quiz-option-wrong {
        background: rgba(239, 68, 68, 0.12);
        border: 1px solid rgba(239, 68, 68, 0.4);
        color: #f87171;
    }
    .quiz-option-neutral {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        color: #94a3b8;
    }

    /* ── Buttons ── */
    .stButton > button {
        border-radius: 10px !important;
        font-weight: 500 !important;
        transition: all 0.2s !important;
        font-size: 0.85rem !important;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        border: none !important;
        color: #ffffff !important;
        box-shadow: 0 2px 8px rgba(99, 102, 241, 0.3) !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
        box-shadow: 0 4px 16px rgba(99, 102, 241, 0.45) !important;
        transform: translateY(-1px) !important;
        color: #ffffff !important;
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        font-size: 0.85rem !important;
        font-weight: 500 !important;
        border-radius: 10px !important;
    }

    /* ── File Uploader ── */
    section[data-testid="stFileUploader"] {
        border-radius: 10px;
    }
    section[data-testid="stFileUploader"] > div {
        border-radius: 10px !important;
    }
    /* Deutsche Texte für File Uploader */
    [data-testid="stFileUploaderDropzone"] span:first-of-type {
        visibility: hidden;
        position: relative;
    }
    [data-testid="stFileUploaderDropzone"] span:first-of-type::after {
        content: "Datei hierher ziehen";
        visibility: visible;
        position: absolute;
        left: 0;
        right: 0;
    }
    [data-testid="stFileUploaderDropzone"] small {
        visibility: hidden;
        position: relative;
    }
    [data-testid="stFileUploaderDropzone"] small::after {
        content: "Max. 200MB pro Datei · PDF";
        visibility: visible;
        position: absolute;
        left: 0;
        right: 0;
    }
    [data-testid="stFileUploaderDropzone"] button {
        visibility: hidden;
        position: relative;
    }
    [data-testid="stFileUploaderDropzone"] button::after {
        content: "Datei auswählen";
        visibility: visible;
        position: absolute;
        left: 50%;
        transform: translateX(-50%);
        white-space: nowrap;
    }

    /* ── Divider ── */
    hr {
        border-color: rgba(255,255,255,0.06) !important;
        margin: 0.8rem 0 !important;
    }

    /* ── Progress Bar ── */
    .stProgress > div > div {
        background: linear-gradient(90deg, #6366f1, #8b5cf6) !important;
        border-radius: 6px !important;
    }

    /* ── Metric hiding default ── */
    [data-testid="stMetric"] {
        display: none;
    }

    /* ── Alert boxes ── */
    .stAlert {
        border-radius: 10px !important;
    }

    /* ── Text Input Tooltip (Press Enter to apply) verstecken ── */
    .stTextInput div[data-baseweb="tooltip"] {
        display: none !important;
    }
    .stTextInput [data-testid="InputInstructions"] {
        display: none !important;
    }

    /* ── Mode Badge ── */
    .mode-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.35rem 1rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        margin-bottom: 0.8rem;
    }
    .mode-badge-chat {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(139, 92, 246, 0.1));
        border: 1px solid rgba(99, 102, 241, 0.3);
        color: #a5b4fc;
    }
    .mode-badge-quiz {
        background: linear-gradient(135deg, rgba(234, 179, 8, 0.15), rgba(245, 158, 11, 0.1));
        border: 1px solid rgba(234, 179, 8, 0.3);
        color: #fbbf24;
    }
    .mode-badge-sparring {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15), rgba(52, 211, 153, 0.1));
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #6ee7b7;
    }

    /* ── Disable auto-anchor links on headings in chat ── */
    [data-testid="stChatMessage"] h1 a,
    [data-testid="stChatMessage"] h2 a,
    [data-testid="stChatMessage"] h3 a,
    [data-testid="stChatMessage"] h4 a {
        display: none !important;
        pointer-events: none !important;
    }

    /* ── Expander overflow fix ── */
    [data-testid="stExpander"] hr {
        display: none !important;
    }
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] {
        overflow: hidden !important;
    }

    /* ── Sidebar-Expander: kompakte Buttons (Schriftgröße passend) ── */
    section[data-testid="stSidebar"] [data-testid="stExpander"] .stButton > button {
        font-size: 0.85rem !important;
        padding: 0.45rem 0.75rem !important;
        min-height: 0 !important;
        line-height: 1.3 !important;
        white-space: nowrap !important;
    }
    section[data-testid="stSidebar"] [data-testid="stExpander"] .stButton > button p {
        font-size: 0.85rem !important;
        margin: 0 !important;
    }

    /* ── Source chunk in expander ── */
    .source-chunk {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 8px;
        padding: 0.8rem;
        margin-bottom: 0.6rem;
        font-size: 0.82rem;
    }
    .source-chunk .source-header {
        font-size: 0.75rem;
        font-weight: 600;
        color: #a5b4fc;
        margin-bottom: 0.3rem;
    }
    .source-chunk .source-text {
        color: #94a3b8;
        font-size: 0.78rem;
        line-height: 1.45;
        white-space: pre-wrap;
        word-wrap: break-word;
    }
</style>
""", unsafe_allow_html=True)

# ── zugangskontrolle ──
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown("""
    <div style="text-align:center; padding:3rem 1rem;">
        <h2>🎓 Kfz-Haftpflicht Lern-Assistent</h2>
        <p style="color:#888; font-size:0.95rem;">
            Prototyp — nur für autorisierte Nutzer.
        </p>
    </div>
    """, unsafe_allow_html=True)
    with st.form("login_form", clear_on_submit=False):
        password = st.text_input("Zugangspasswort", type="password", placeholder="Passwort eingeben...")
        submitted = st.form_submit_button("Zugang anfordern", use_container_width=True)
        if submitted:
            if password == "ragsys1":
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Falsches Passwort")
    st.stop()

# ── prototyp-hinweis ──
# embeddings laufen immer ueber openai
# der chat-pfad ist variabel das spiegelt der hinweistext wider
def _privacy_notice() -> str:
    base = (
        "**Prototyp** — Diese Anwendung dient ausschließlich zu Lern- "
        "und Demonstrationszwecken."
    )
    parts = ["Embeddings über OpenAI"]
    if st.session_state.get("user_api_key"):
        provider = st.session_state.get("user_provider", "API")
        parts.append(f"Chat-Anfragen über {provider}")
    else:
        parts.append("Chat-Anfragen über lokales Ollama")
    return (
        base
        + " Hinweis: "
        + " und ".join(parts)
        + " — beachte die Datenschutzerklärung des jeweiligen Anbieters."
    )

st.info(_privacy_notice(), icon="🔬")

# ── datenbank initialisieren ──
@st.cache_resource
def setup_database():
    init_db()
    return True

db_error = None
try:
    setup_database()
    db_connected = True
except Exception as e:
    db_connected = False
    db_error = str(e)

# ── session state initialisieren ──
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "session_id" not in st.session_state:
    if db_connected:
        st.session_state.session_id = create_chat_session()
    else:
        st.session_state.session_id = None
if "mode" not in st.session_state:
    st.session_state.mode = None
if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = None
if "quiz_answered" not in st.session_state:
    st.session_state.quiz_answered = False
if "quiz_score" not in st.session_state:
    st.session_state.quiz_score = {"correct": 0, "total": 0}
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "user_api_key" not in st.session_state:
    st.session_state.user_api_key = ""
if "user_provider" not in st.session_state:
    st.session_state.user_provider = "openai"
if "current_chat_id" not in st.session_state:
    st.session_state.current_chat_id = None

# ── first-run-wizard ──
# wenn die erstkonfig noch nicht durch ist uebernimmt der wizard
# den hauptbereich und der rest des skripts wird gestoppt
if db_connected and not _setup_completed():
    _render_setup_wizard()
    st.stop()


# reranker im hintergrund vorladen damit der erste retrieve schnell ist
_warmup_reranker_async()


# ── juengsten chat aus localStorage wiederherstellen ──
# nur wenn chat-history leer ist (also der user gerade in den mode eingestiegen ist)
# das streamlit-local-storage-component braucht ein bis zwei reruns
def _restore_latest_chat_if_needed():
    mode = st.session_state.get("mode")
    if mode not in ("Chat", "Sparring"):
        return
    flag = f"_ls_restored_{mode}"
    if st.session_state.get(flag):
        return
    if st.session_state.chat_history:
        st.session_state[flag] = True
        return
    chats = _load_chats(mode)
    if chats:
        latest = chats[0]
        st.session_state.chat_history = latest.get("messages", [])
        st.session_state.current_chat_id = latest.get("id")
        st.session_state[flag] = True


_restore_latest_chat_if_needed()

# ── sidebar ──
with st.sidebar:
    # branding-button klickbar zur startseite
    if st.button("🎓\nKfz-Haftpflicht\nLern-Assistent", use_container_width=True, key="brand_home"):
        st.session_state.mode = None
        st.session_state.chat_history = []
        st.session_state.session_id = create_chat_session()
        st.session_state.current_chat_id = None
        st.session_state.quiz_data = None
        st.session_state.quiz_answered = False
        st.session_state.quiz_score = {"correct": 0, "total": 0}
        st.session_state.pop("prefetch_started_for", None)
        # localStorage bleibt
        # restore-flags clearen damit beim naechsten mode-aufruf der juengste chat geladen wird
        st.session_state.pop("_ls_restored_Chat", None)
        st.session_state.pop("_ls_restored_Sparring", None)
        st.rerun()

    st.divider()

    # modus-auswahl
    modes = ["Chat", "Quiz", "Sparring"]
    current_index = modes.index(st.session_state.mode) if st.session_state.mode in modes else None
    new_mode = st.radio(
        "Modus",
        modes,
        horizontal=True,
        label_visibility="collapsed",
        index=current_index
    )

    # bei moduswechsel chat-verlauf zuruecksetzen
    if new_mode is not None and new_mode != st.session_state.mode:
        st.session_state.mode = new_mode
        st.session_state.chat_history = []
        st.session_state.current_chat_id = None
        st.session_state.session_id = create_chat_session()
        st.session_state.quiz_data = None
        st.session_state.quiz_answered = False
        st.session_state.quiz_score = {"correct": 0, "total": 0}
        st.session_state.pop("prefetch_started_for", None)
        # restore-flag des neuen mode loeschen
        # damit der juengste chat aus localStorage wiederhergestellt wird
        st.session_state.pop(f"_ls_restored_{new_mode}", None)
        st.rerun()

    if st.button("＋ Neuer Chat", use_container_width=True):
        _current_mode = st.session_state.mode
        st.session_state.chat_history = []
        st.session_state.current_chat_id = None
        st.session_state.session_id = create_chat_session()
        st.session_state.quiz_data = None
        st.session_state.quiz_answered = False
        st.session_state.quiz_score = {"correct": 0, "total": 0}
        st.session_state.pop("prefetch_started_for", None)
        # localStorage bleibt - vergangene chats sind weiter da
        # restore-flag setzen damit der juengste chat nicht direkt wieder geladen wird
        if _current_mode in ("Chat", "Sparring"):
            st.session_state[f"_ls_restored_{_current_mode}"] = True
        st.rerun()

    # ── vergangene chats (nur in chat- und sparring-modus) ──
    # in einem expander damit das kompakte sidebar-expander-styling greift
    if st.session_state.mode in ("Chat", "Sparring"):
        _past_chats = _load_chats(st.session_state.mode)
        if _past_chats:
            with st.expander(
                f"Vergangene Chats ({len(_past_chats)})",
                expanded=True,
            ):
                for _chat in _past_chats:
                    _cid = _chat.get("id", "")
                    _title = _chat.get("title", "Chat") or "Chat"
                    if len(_title) > 28:
                        _title = _title[:25] + "…"
                    _is_active = _cid == st.session_state.current_chat_id
                    _prefix = "▶ " if _is_active else ""
                    col_a, col_b = st.columns([5, 1])
                    with col_a:
                        if st.button(
                            f"{_prefix}{_title}",
                            key=f"loadchat_{_cid}",
                            use_container_width=True,
                            disabled=_is_active,
                        ):
                            st.session_state.chat_history = _chat.get("messages", [])
                            st.session_state.current_chat_id = _cid
                            st.session_state.session_id = create_chat_session()
                            # restore-flag setzen damit der juengste nicht ueberschreibt
                            st.session_state[f"_ls_restored_{st.session_state.mode}"] = True
                            st.rerun()
                    with col_b:
                        if st.button(
                            "✕",
                            key=f"delchat_{_cid}",
                            help="Chat löschen",
                        ):
                            _delete_chat(st.session_state.mode, _cid)
                            if _is_active:
                                st.session_state.chat_history = []
                                st.session_state.current_chat_id = None
                            st.rerun()

    st.divider()

    # ── embedding-status (read-only) ──
    # schluessel wurde im setup-wizard hinterlegt hier nur info-zeile
    st.markdown(
        '<div class="sidebar-section">Embeddings</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Aktiv: <b>OpenAI text-embedding-3-small</b><br/>"
        "Schlüssel im Setup hinterlegt — über Erweitert ▶ Reset änderbar.",
        unsafe_allow_html=True,
    )

    st.divider()

    # ── chat-anbieter ──
    has_user_key = bool(st.session_state.user_api_key)
    label = (
        f"Chat-Anbieter aktiv ({st.session_state.user_provider})"
        if has_user_key else "Chat-Anbieter wählen (Ollama lokal als Default)"
    )
    # expander-zustand:
    # beim allerersten laden ohne key einmalig offen
    # sonst zu - klick auf header oeffnet manuell
    # bleibt offen wenn innerhalb interagiert wird (via on_change-callback)
    # schliesst automatisch bei aktionen ausserhalb
    # nach uebernehmen zwingend zu
    if "apikey_expander_initial_done" not in st.session_state:
        st.session_state.apikey_expander_initial_done = True
        _initial_open = not has_user_key
    else:
        _initial_open = False

    _keep_open = st.session_state.pop("apikey_keep_open", False)
    _force_collapse = st.session_state.pop("apikey_just_applied", False)

    if _force_collapse:
        _expander_open = False
    elif _keep_open:
        _expander_open = True
    else:
        _expander_open = _initial_open

    def _hold_apikey_expander_open():
        """haelt den expander beim naechsten rerun offen
        wird per on_change/on_click an die widgets innen gehaengt
        damit der user dort weiter arbeiten kann ohne dass der expander zuklappt
        """
        st.session_state.apikey_keep_open = True

    with st.expander(label, expanded=_expander_open):
        if has_user_key:
            st.caption(
                "Chat-Antworten werden über deinen Schlüssel bei "
                f"<b>{st.session_state.user_provider}</b> generiert.",
                unsafe_allow_html=True,
            )
        else:
            st.caption(
                "Ohne Schlüssel wird auf eine lokale Ollama-Instanz "
                "zurückgegriffen, kostenlos aber Ollama muss installiert "
                "und gestartet sein (https://ollama.com). Das Modell "
                "`llama3.2:3b` (~2,0 GB) wird beim ersten Aufruf "
                "automatisch heruntergeladen. Falls der Auto-Pull "
                "scheitert, einmalig im Terminal: `ollama pull llama3.2:3b`. "
                "Schneller und ohne Download geht es mit einem kostenlosen "
                "Groq-Schlüssel (https://groq.com)."
            )

        # radio-optionen inkl expliziter "kein schluessel"-option
        _provider_options = ["none", "openai", "groq", "gemini"]
        _provider_labels = {
            "none": "Kein Schlüssel (Ollama lokal)",
            "openai": "OpenAI",
            "groq": "Groq (kostenlos)",
            "gemini": "Google Gemini",
        }
        # vorauswahl: bei vorhandenem schluessel der gespeicherte provider
        # sonst "kein schluessel"
        if has_user_key and st.session_state.user_provider in _provider_options:
            _initial_idx = _provider_options.index(st.session_state.user_provider)
        else:
            _initial_idx = 0  # "none"

        provider_choice = st.radio(
            "Anbieter",
            options=_provider_options,
            format_func=lambda x: _provider_labels[x],
            index=_initial_idx,
            key="provider_select",
            on_change=_hold_apikey_expander_open,
        )

        # api-key-feld nur wenn ein provider mit schluessel gewaehlt ist
        if provider_choice == "none":
            key_input = ""
            st.caption(
                "Ollama wird verwendet. Kein Schlüssel nötig. "
                "Klicke auf <b>Übernehmen</b>, um die Auswahl zu speichern.",
                unsafe_allow_html=True,
            )
        else:
            key_input = st.text_input(
                "API-Schlüssel",
                value=st.session_state.user_api_key,
                type="password",
                placeholder="sk-... / gsk_... / AIzaSy...",
                help="Wird nur in deiner Browser-Sitzung gespeichert, nicht serverseitig.",
                key="api_key_input",
                on_change=_hold_apikey_expander_open,
            )

        if st.button("Übernehmen", use_container_width=True, key="apply_key",
                     type="primary", on_click=_hold_apikey_expander_open):
            if provider_choice == "none":
                st.session_state.user_api_key = ""
                st.session_state.user_provider = "openai"  # neutraler Default
                st.session_state.apikey_just_applied = True
                st.rerun()
            else:
                stripped = key_input.strip()
                if stripped:
                    st.session_state.user_api_key = stripped
                    st.session_state.user_provider = provider_choice
                    st.session_state.apikey_just_applied = True
                    st.rerun()
                else:
                    # bei validierungsfehler KEIN st.rerun()
                    # sonst wird die warnung sofort weggerendert
                    # expander bleibt offen warnung steht
                    st.warning(
                        "Kein Schlüssel eingegeben — bitte ausfüllen oder "
                        'die Option „Kein Schlüssel (Ollama lokal)" wählen.'
                    )

    st.divider()

    if not db_connected:
        st.error(f"Datenbankverbindung fehlgeschlagen: {db_error}")
        st.info("Starte PostgreSQL mit: `docker-compose up -d`")
        st.stop()

    # dokumente anzeigen
    documents = get_all_documents()
    chunk_count = get_chunk_count()

    # stats
    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-card">
            <div class="stat-value">{len(documents)}</div>
            <div class="stat-label">Dokumente</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{chunk_count:,}</div>
            <div class="stat-label">Chunks</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # dokumentenliste
    st.markdown('<div class="sidebar-section">Wissensbasis</div>', unsafe_allow_html=True)

    if documents:
        for doc in documents:
            col1, col2 = st.columns([6, 1])
            with col1:
                short_name = doc['filename']
                if len(short_name) > 30:
                    short_name = short_name[:27] + "..."
                st.markdown(
                    f"<div style='font-size:0.82rem;color:#cbd5e1;padding:0.3rem 0;'>"
                    f"📄 {short_name}<br>"
                    f"<span style='font-size:0.7rem;color:#64748b;'>{doc['page_count']} Seiten</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            with col2:
                st.write("")
                if st.button("✕", key=f"del_{doc['id']}", help=f"{doc['filename']} entfernen"):
                    remove_document(doc["filename"])
                    st.rerun()
    else:
        st.caption("Noch keine Dokumente hochgeladen.")

    st.divider()

    # pdf upload
    st.markdown('<div class="sidebar-section">Dokument hinzufügen</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "PDF hochladen",
        type=["pdf"],
        help="PDF-Datei zur Wissensbasis hinzufügen",
        label_visibility="collapsed",
        key=f"uploader_{st.session_state.uploader_key}"
    )

    if uploaded_file is not None:
        if st.button("Verarbeiten", type="primary", use_container_width=True):
            os.makedirs(DOCUMENTS_DIR, exist_ok=True)
            filepath = os.path.join(DOCUMENTS_DIR, uploaded_file.name)
            with open(filepath, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner(f"Verarbeite {uploaded_file.name}..."):
                result = ingest_document(filepath, uploaded_file.name)

            if result["status"] == "ingested":
                st.success(
                    f"{result['filename']}: "
                    f"{result['pages']} Seiten, {result['chunks']} Chunks"
                )
            elif result["status"] == "unchanged":
                st.info("Dokument bereits vorhanden.")
            else:
                st.error(f"Fehler: {result.get('message', 'Unbekannt')}")
            # uploader zuruecksetzen
            st.session_state.uploader_key += 1
            st.rerun()

    st.divider()

    # ── debug: latenzen der letzten anfrage ──
    # nur sichtbar wenn schon eine anfrage durchgelaufen ist
    _last_t = st.session_state.get("last_retrieval_timings")
    if _last_t is not None:
        with st.expander("⏱ Letzte Anfrage", expanded=False):
            st.caption(
                f"Gesamt: <b>{_last_t.total_ms:.0f} ms</b>"
                f"&nbsp;·&nbsp; {_last_t.num_queries} Query-Variante(n)"
                f"&nbsp;·&nbsp; {_last_t.num_fused} Treffer nach RRF"
                + ("&nbsp;·&nbsp; parallel" if _last_t.parallel else ""),
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div style='font-size:0.78rem;color:#94a3b8;line-height:1.6'>
                Multi-Query: <b>{_last_t.multi_query_ms:.0f} ms</b><br/>
                Hybrid-Search: <b>{_last_t.search_ms:.0f} ms</b><br/>
                RRF: <b>{_last_t.fusion_ms:.0f} ms</b><br/>
                Reranker: <b>{_last_t.rerank_ms:.0f} ms</b>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── erweitert: wissensbasis zuruecksetzen ──
    # noetig bei key-wechsel oder wenn man komplett neu anfangen will
    # zwei-klick-confirm gegen versehentliches ausloesen
    with st.expander("⚙ Erweitert", expanded=False):
        st.caption(
            "Setzt alle Chunks, Dokumente, Chat-Verläufe und die "
            "Setup-Konfiguration zurück. Beim nächsten App-Start läuft "
            "die Erstkonfiguration erneut."
        )
        if not st.session_state.get("confirm_reset"):
            if st.button(
                "Wissensbasis zurücksetzen",
                key="reset_request",
                use_container_width=True,
            ):
                st.session_state.confirm_reset = True
                st.rerun()
        else:
            st.warning(
                "Wirklich alle Daten löschen? Dieser Schritt kann nicht "
                "rückgängig gemacht werden."
            )
            cc1, cc2 = st.columns(2)
            with cc1:
                if st.button(
                    "Ja, zurücksetzen",
                    type="primary",
                    use_container_width=True,
                    key="reset_confirm",
                ):
                    reset_knowledge_base()
                    reset_provider_caches()
                    # alle chats fuer beide modi loeschen
                    _clear_chats("Chat")
                    _clear_chats("Sparring")
                    # session-state leeren damit der wizard frisch startet
                    for k in (
                        "confirm_reset", "wizard_step",
                        "user_api_key", "user_provider",
                        "chat_history", "session_id", "current_chat_id",
                        "quiz_data", "quiz_answered", "quiz_score",
                        "mode", "apikey_expander_initial_done",
                        "prefetch_started_for", "last_retrieval_timings",
                        "_ls_restored_Chat", "_ls_restored_Sparring",
                    ):
                        st.session_state.pop(k, None)
                    st.rerun()
            with cc2:
                if st.button(
                    "Abbrechen",
                    use_container_width=True,
                    key="reset_cancel",
                ):
                    st.session_state.confirm_reset = False
                    st.rerun()


# ── hauptbereich ──

# ── startscreen kein modus ausgewaehlt ──
if st.session_state.mode is None:
    st.markdown("""
    <div class="main-header">
        <span class="header-icon">🎓</span>
        <h1>Kfz-Haftpflichtversicherung</h1>
        <div class="header-sub">Dein KI-Lern-Assistent</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="welcome-grid">
        <div class="welcome-card">
            <span class="card-icon">💬</span>
            <div class="card-title">Chat-Modus</div>
            <div class="card-desc">Stelle gezielte Fragen und erhalte präzise Antworten mit Quellenangaben.</div>
        </div>
        <div class="welcome-card">
            <span class="card-icon">🧠</span>
            <div class="card-title">Quiz-Modus</div>
            <div class="card-desc">Teste dein Wissen mit Multiple-Choice-Fragen, die direkt aus deinen Materialien generiert werden.</div>
        </div>
        <div class="welcome-card">
            <span class="card-icon">⚡</span>
            <div class="card-title">Sparring-Modus</div>
            <div class="card-desc">Sokratisches Lernen — erarbeite Antworten selbst durch gezielte Gegenfragen des Assistenten.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.info("Wähle einen Modus in der Sidebar, um loszulegen.")

    if chunk_count == 0:
        st.warning(
            "Die Wissensbasis ist leer. Bitte lade PDFs hoch, um den Assistenten nutzen zu können."
        )

# ── chat-modus ──
elif st.session_state.mode == "Chat":

    st.markdown('<div class="mode-badge mode-badge-chat">💬 Chat-Modus</div>', unsafe_allow_html=True)

    # welcome screen wenn kein chat-verlauf
    if not st.session_state.chat_history:
        st.markdown("""
        <div class="main-header">
            <span class="header-icon">💬</span>
            <h1>Chat-Modus</h1>
            <div class="header-sub">Stelle Fragen und erhalte fundierte Antworten</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="welcome-grid">
            <div class="welcome-card">
                <span class="card-icon">📖</span>
                <div class="card-title">Quellenbasiert</div>
                <div class="card-desc">Antworten basieren ausschließlich auf den hochgeladenen Materialien.</div>
            </div>
            <div class="welcome-card">
                <span class="card-icon">📄</span>
                <div class="card-title">Mit Quellenangaben</div>
                <div class="card-desc">Jede Antwort enthält die genauen Quellen mit Dateiname und Seitenzahl zum Nachschlagen.</div>
            </div>
            <div class="welcome-card">
                <span class="card-icon">💡</span>
                <div class="card-title">Verständlich erklärt</div>
                <div class="card-desc">Juristische Fachbegriffe werden studierendengerecht aufbereitet und verständlich erklärt.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if chunk_count == 0:
            st.warning(
                "Die Wissensbasis ist leer. Bitte lade PDFs hoch, um den Assistenten nutzen zu können."
            )

    # chat-verlauf anzeigen mit den gespeicherten quellen pro nachricht
    for idx, msg in enumerate(st.session_state.chat_history):
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🎓"):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("Verwendete Quellen anzeigen"):
                    _render_chunk_sources(msg["sources"], key_prefix=f"chat_{idx}")

    # chat-eingabe
    if prompt := st.chat_input("Stelle eine Frage zur Kfz-Haftpflichtversicherung..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🎓"):
            # ist es eine begruessung/smalltalk?
            if is_greeting(prompt):
                answer = generate_greeting_response(
                    prompt,
                    st.session_state.chat_history[:-1]
                )
                context_chunks = []
                st.markdown(answer)
            else:
                with st.spinner("Suche in der Wissensbasis..."):
                    standalone_query = rewrite_question_with_history(
                        st.session_state.chat_history[:-1],
                        prompt
                    )
                    context_chunks = retrieve(standalone_query)
                    # latenzen fuer den debug-expander in der sidebar persistieren
                    _t = get_last_timings()
                    if _t is not None:
                        st.session_state.last_retrieval_timings = _t

                if not context_chunks:
                    answer = (
                        "Leider konnte ich keine relevanten Informationen in der "
                        "Wissensbasis finden. Bitte formuliere deine Frage anders "
                        "oder stelle sicher, dass die relevanten Dokumente hochgeladen sind."
                    )
                    st.markdown(answer)
                else:
                    answer = st.write_stream(
                        generate_answer_stream(
                            prompt,
                            context_chunks,
                            st.session_state.chat_history[:-1],
                        )
                    )

        # quellen mit der assistant-nachricht persistieren
        # der naechste rerun rendert sie aus der history
        # damit die pdf-sprung-buttons stabil klickbar bleiben
        answer_lower = answer.lower()
        show_sources = bool(context_chunks) and not any(h in answer_lower for h in NO_ANSWER_HINTS)
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": context_chunks if show_sources else None,
        })
        if st.session_state.session_id:
            save_chat_message(st.session_state.session_id, "user", prompt)
            save_chat_message(st.session_state.session_id, "assistant", answer)
        if st.session_state.current_chat_id is None:
            st.session_state.current_chat_id = _new_chat_id()
        _save_chat(
            st.session_state.mode,
            st.session_state.current_chat_id,
            st.session_state.chat_history,
        )
        st.rerun()

# ── quiz-modus ──
elif st.session_state.mode == "Quiz":
    st.markdown('<div class="mode-badge mode-badge-quiz">🧠 Quiz-Modus</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="main-header">
        <span class="header-icon">🧠</span>
        <h1>Quiz-Modus</h1>
        <div class="header-sub">Teste dein Wissen zur Kfz-Haftpflichtversicherung</div>
    </div>
    """, unsafe_allow_html=True)

    # welcome-karten wenn noch keine frage generiert
    if not st.session_state.quiz_data and st.session_state.quiz_score["total"] == 0:
        st.markdown("""
        <div class="welcome-grid">
            <div class="welcome-card">
                <span class="card-icon">🎯</span>
                <div class="card-title">Multiple-Choice</div>
                <div class="card-desc">Beantworte Fragen mit vier Antwortmöglichkeiten — generiert aus deinen Materialien.</div>
            </div>
            <div class="welcome-card">
                <span class="card-icon">📊</span>
                <div class="card-title">Fortschritt</div>
                <div class="card-desc">Verfolge deinen Score und sieh, wie viele Fragen du bereits richtig beantwortet hast.</div>
            </div>
            <div class="welcome-card">
                <span class="card-icon">🔍</span>
                <div class="card-title">Themen wählen</div>
                <div class="card-desc">Gib ein Thema ein oder lass zufällig Fragen aus allen Bereichen generieren.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # score anzeigen
    score = st.session_state.quiz_score
    if score["total"] > 0:
        pct = score["correct"] / score["total"]
        st.markdown(f"""
        <div class="score-badge">
            🏆 {score['correct']} / {score['total']} richtig ({pct:.0%})
        </div>
        """, unsafe_allow_html=True)
        st.progress(pct)

    # quiz-thema
    col1, col2 = st.columns([3, 1])
    with col1:
        quiz_topic = st.text_input(
            "Thema (optional)",
            placeholder="z.B. Halterhaftung, Regress, Deckungsumfang...",
            label_visibility="collapsed",
            key="quiz_topic_input"
        )
    with col2:
        generate_btn = st.button("Neue Frage", type="primary", use_container_width=True)

    # enter im textfeld oder button-klick generiert neue frage
    topic_submitted = quiz_topic and quiz_topic != st.session_state.get("last_quiz_topic", "")
    if generate_btn or topic_submitted:
        if topic_submitted:
            st.session_state.last_quiz_topic = quiz_topic

        # check ob im hintergrund schon eine frage vorgeneriert wurde
        # bis zu 4 sekunden warten falls der thread noch laeuft
        # passt das prefetched-topic nicht oder thread crashed: synchron generieren
        prefetched_data = None
        current_topic = quiz_topic if quiz_topic else None
        if st.session_state.session_id is not None:
            with st.spinner("Naechste Frage wird geladen..."):
                popped = _pop_prefetched_quiz(
                    st.session_state.session_id, current_topic, wait_ms=4000
                )
            if popped is not None:
                data, _err = popped
                if data is not None:
                    prefetched_data = data
        st.session_state.pop("prefetch_started_for", None)

        if prefetched_data is not None:
            st.session_state.quiz_data = prefetched_data
            st.session_state.quiz_answered = False
        else:
            with st.spinner("Generiere Frage..."):
                quiz_data, quiz_error = generate_quiz_question(current_topic)
                if quiz_data:
                    st.session_state.quiz_data = quiz_data
                    st.session_state.quiz_answered = False
                else:
                    st.error(
                        quiz_error
                        or "Konnte keine Frage generieren. Bitte erneut versuchen."
                    )

    # frage anzeigen
    if st.session_state.quiz_data:
        qd = st.session_state.quiz_data

        st.markdown(f"""
        <div class="quiz-question-card">
            <div class="question-text">{qd['question']}</div>
            <div class="question-source">Quelle: {qd.get('source', 'eigene Materialien')}</div>
        </div>
        """, unsafe_allow_html=True)

        letters = ["A", "B", "C", "D"]

        if not st.session_state.quiz_answered:
            for i, option in enumerate(qd["options"]):
                label = f"{letters[i]}.  {option}" if i < len(letters) else option
                if st.button(
                    label,
                    key=f"quiz_option_{i}",
                    use_container_width=True
                ):
                    st.session_state.quiz_answered = True
                    st.session_state.quiz_score["total"] += 1

                    if i == qd["correct_index"]:
                        st.session_state.selected_correct = True
                        st.session_state.quiz_score["correct"] += 1
                    else:
                        st.session_state.selected_correct = False
                    st.session_state.selected_index = i
                    st.rerun()
        else:
            for i, option in enumerate(qd["options"]):
                letter = letters[i] if i < len(letters) else ""
                safe_option = html.escape(option)
                if i == qd["correct_index"]:
                    icon = "✅"
                    css_class = "quiz-option-correct"
                elif i == st.session_state.get("selected_index"):
                    icon = "❌"
                    css_class = "quiz-option-wrong"
                else:
                    icon = ""
                    css_class = "quiz-option-neutral"
                st.markdown(f"""
                <div class="quiz-option-answered {css_class}">
                    <span class="option-letter">{letter}.</span>
                    <span>{safe_option}</span>
                    <span style="margin-left:auto;">{icon}</span>
                </div>
                """, unsafe_allow_html=True)

            if st.session_state.get("selected_correct"):
                st.balloons()
                st.success("Richtig!")
            else:
                st.error("Leider falsch.")

            st.info(f"**Erklärung:** {qd['explanation']}")

            # pdf-sprung-buttons zur verifikation der antwort
            quiz_sources = qd.get("source_chunks") or []
            if quiz_sources:
                st.markdown("**Antwort im Skript nachlesen:**")
                for i, src in enumerate(quiz_sources, 1):
                    pdf_path = os.path.join(DOCUMENTS_DIR, src["filename"])
                    if os.path.exists(pdf_path):
                        if st.button(
                            f"📖 {src['filename']} — Seite {src['page_number']}",
                            key=f"pdf_quiz_{i}_{src['page_number']}",
                            use_container_width=True,
                        ):
                            _show_pdf_dialog(
                                pdf_path,
                                src["page_number"],
                                src["filename"],
                            )

            # naechste frage im hintergrund vorbereiten
            # nur einmal pro answered-zustand starten
            _topic_now = quiz_topic if quiz_topic else None
            _started_for = st.session_state.get("prefetch_started_for")
            if (
                st.session_state.session_id is not None
                and _started_for != _topic_now
            ):
                st.session_state.prefetch_started_for = _topic_now
                _prefetch_next_quiz(st.session_state.session_id, _topic_now)

# ── sparring-modus (sokratisch) ──
elif st.session_state.mode == "Sparring":

    st.markdown('<div class="mode-badge mode-badge-sparring">⚡ Sparring-Modus</div>', unsafe_allow_html=True)

    # welcome screen wenn kein chat-verlauf
    if not st.session_state.chat_history:
        st.markdown("""
        <div class="main-header">
            <span class="header-icon">⚡</span>
            <h1>Sparring-Modus</h1>
            <div class="header-sub">Sokratisches Lernen — erarbeite dir die Antworten selbst</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="welcome-grid">
            <div class="welcome-card">
                <span class="card-icon">🤔</span>
                <div class="card-title">Gegenfragen</div>
                <div class="card-desc">Statt direkter Antworten bekommst du gezielte Gegenfragen, die dich zum Nachdenken anregen.</div>
            </div>
            <div class="welcome-card">
                <span class="card-icon">🧩</span>
                <div class="card-title">Schritt für Schritt</div>
                <div class="card-desc">Der Assistent führt dich durch Teilfragen zur vollständigen Antwort — wie ein Tutor.</div>
            </div>
            <div class="welcome-card">
                <span class="card-icon">🏆</span>
                <div class="card-title">Tiefes Verständnis</div>
                <div class="card-desc">Durch eigenständiges Erarbeiten bleiben die Inhalte nachhaltig im Gedächtnis.</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # chat-verlauf anzeigen mit den gespeicherten quellen pro nachricht
    for idx, msg in enumerate(st.session_state.chat_history):
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "⚡"):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("Verwendete Quellen anzeigen"):
                    _render_chunk_sources(msg["sources"], key_prefix=f"sparring_{idx}")

    # chat-eingabe
    if prompt := st.chat_input("Stelle eine Frage — ich helfe dir, die Antwort selbst zu finden..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="⚡"):
            if is_greeting(prompt):
                answer = generate_greeting_response(
                    prompt,
                    st.session_state.chat_history[:-1]
                )
                context_chunks = []
                st.markdown(answer)
            else:
                with st.spinner("Denke nach..."):
                    standalone_query = rewrite_question_with_history(
                        st.session_state.chat_history[:-1],
                        prompt
                    )
                    context_chunks = retrieve(standalone_query)
                    _t = get_last_timings()
                    if _t is not None:
                        st.session_state.last_retrieval_timings = _t

                if not context_chunks:
                    answer = (
                        "Leider konnte ich keine relevanten Informationen in der "
                        "Wissensbasis finden. Bitte formuliere deine Frage anders "
                        "oder stelle sicher, dass die relevanten Dokumente hochgeladen sind."
                    )
                    st.markdown(answer)
                else:
                    answer = st.write_stream(
                        generate_sparring_stream(
                            prompt,
                            context_chunks,
                            st.session_state.chat_history[:-1],
                        )
                    )

        answer_lower = answer.lower()
        show_sources = bool(context_chunks) and not any(h in answer_lower for h in NO_ANSWER_HINTS)
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": context_chunks if show_sources else None,
        })
        if st.session_state.session_id:
            save_chat_message(st.session_state.session_id, "user", prompt)
            save_chat_message(st.session_state.session_id, "assistant", answer)
        if st.session_state.current_chat_id is None:
            st.session_state.current_chat_id = _new_chat_id()
        _save_chat(
            st.session_state.mode,
            st.session_state.current_chat_id,
            st.session_state.chat_history,
        )
        st.rerun()
