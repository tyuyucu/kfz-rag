import os
import html
import streamlit as st

# Streamlit Cloud: Secrets in os.environ laden BEVOR andere Module importiert werden
try:
    for _key, _val in st.secrets.items():
        if isinstance(_val, str) and _key not in os.environ:
            os.environ[_key] = _val
except Exception:
    pass

from db.database import init_db, get_all_documents, create_chat_session, save_chat_message
from db.vector_store import get_chunk_count
from ingestion.pipeline import ingest_document, remove_document
from retrieval.pipeline import retrieve
from generation.history import rewrite_question_with_history
from generation.generator import generate_answer, is_greeting, generate_greeting_response, generate_sparring_response
from generation.quiz import generate_quiz_question
from config import DOCUMENTS_DIR

# ── Konstanten ──
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

# ── Seiten-Konfiguration ──
st.set_page_config(
    page_title="Kfz-Haftpflicht Lern-Assistent",
    page_icon="🎓",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ──
st.markdown("""
<style>
    /* ── Allgemein ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* ── Dark-Mode erzwingen (auch wenn Streamlit-Theme "Light" gewählt wird) ── */
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
    /* Texte, die Streamlit standardmäßig in Light-Mode dunkel rendert */
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown span,
    label, .stRadio label, .stCheckbox label,
    [data-testid="stWidgetLabel"], [data-testid="stMarkdownContainer"] {
        color: #e2e8f0 !important;
    }
    /* Inputs immer dunkel */
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
    /* File Uploader Drop Zone */
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploaderDropzone"] {
        background-color: #1a1f2e !important;
        border-color: rgba(255,255,255,0.1) !important;
    }
    [data-testid="stFileUploader"] section *,
    [data-testid="stFileUploaderDropzone"] * {
        color: #e2e8f0 !important;
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

    /* ── Dokument-Entfernen-Button rot bei Hover ── */
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
        box-shadow: 0 2px 8px rgba(99, 102, 241, 0.3) !important;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 4px 16px rgba(99, 102, 241, 0.45) !important;
        transform: translateY(-1px) !important;
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
    }
</style>
""", unsafe_allow_html=True)

# ── Zugangskontrolle ──
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.markdown("""
    <div style="text-align:center; padding:3rem 1rem;">
        <h2>🎓 Kfz-Haftpflicht Lern-Assistent</h2>
        <p style="color:#888; font-size:0.95rem;">
            Prototyp im Rahmen einer Bachelorarbeit — nur für autorisierte Nutzer.
        </p>
    </div>
    """, unsafe_allow_html=True)
    with st.form("login_form", clear_on_submit=False):
        password = st.text_input("Zugangspasswort", type="password", placeholder="Passwort eingeben...")
        submitted = st.form_submit_button("Zugang anfordern", use_container_width=True)
        if submitted:
            if password == "kfzragsys1":
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Falsches Passwort")
    st.stop()

# ── Prototyp-Hinweis ──
st.info(
    "**Prototyp** — Diese Anwendung wurde im Rahmen einer Bachelorarbeit entwickelt "
    "und dient ausschließlich zu Lern- und Demonstrationszwecken. "
    "Eingaben werden zur Verarbeitung an OpenAI übermittelt.",
    icon="🔬"
)

# ── Datenbank initialisieren ──
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

# ── Session State initialisieren ──
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

# ── Sidebar ──
with st.sidebar:
    # Branding-Button (klickbar → Startseite)
    if st.button("🎓\nKfz-Haftpflicht\nLern-Assistent", use_container_width=True, key="brand_home"):
        st.session_state.mode = None
        st.session_state.chat_history = []
        st.session_state.session_id = create_chat_session()
        st.session_state.quiz_data = None
        st.session_state.quiz_answered = False
        st.session_state.quiz_score = {"correct": 0, "total": 0}
        st.rerun()

    st.divider()

    # Modus-Auswahl
    modes = ["Chat", "Quiz", "Sparring"]
    current_index = modes.index(st.session_state.mode) if st.session_state.mode in modes else None
    new_mode = st.radio(
        "Modus",
        modes,
        horizontal=True,
        label_visibility="collapsed",
        index=current_index
    )

    # Bei Moduswechsel Chat-Verlauf zurücksetzen
    if new_mode is not None and new_mode != st.session_state.mode:
        st.session_state.mode = new_mode
        st.session_state.chat_history = []
        st.session_state.session_id = create_chat_session()
        st.session_state.quiz_data = None
        st.session_state.quiz_answered = False
        st.session_state.quiz_score = {"correct": 0, "total": 0}
        st.rerun()

    if st.button("＋ Neuer Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.session_id = create_chat_session()
        st.session_state.quiz_data = None
        st.session_state.quiz_answered = False
        st.session_state.quiz_score = {"correct": 0, "total": 0}
        st.rerun()

    st.divider()

    if not db_connected:
        st.error(f"Datenbankverbindung fehlgeschlagen: {db_error}")
        st.info("Starte PostgreSQL mit: `docker-compose up -d`")
        st.stop()

    # Dokumente anzeigen
    documents = get_all_documents()
    chunk_count = get_chunk_count()

    # Stats
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

    # Dokumentenliste
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

    # PDF Upload
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
            # Uploader zurücksetzen
            st.session_state.uploader_key += 1
            st.rerun()


# ── Hauptbereich ──

# ── Startscreen (kein Modus ausgewählt) ──
if st.session_state.mode is None:
    st.markdown("""
    <div class="main-header">
        <span class="header-icon">🎓</span>
        <h1>Kfz-Haftpflichtversicherung</h1>
        <div class="header-sub">Dein KI-Lern-Assistent für die Vorlesung</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="welcome-grid">
        <div class="welcome-card">
            <span class="card-icon">💬</span>
            <div class="card-title">Chat-Modus</div>
            <div class="card-desc">Stelle gezielte Fragen zu Themen aus der Vorlesung und erhalte präzise Antworten mit Quellenangaben.</div>
        </div>
        <div class="welcome-card">
            <span class="card-icon">🧠</span>
            <div class="card-title">Quiz-Modus</div>
            <div class="card-desc">Teste dein Wissen mit Multiple-Choice-Fragen, die direkt aus den Vorlesungsmaterialien generiert werden.</div>
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

# ── Chat-Modus ──
elif st.session_state.mode == "Chat":

    st.markdown('<div class="mode-badge mode-badge-chat">💬 Chat-Modus</div>', unsafe_allow_html=True)

    # Welcome Screen wenn kein Chat-Verlauf
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

    # Chat-Verlauf anzeigen
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "🎓"):
            st.markdown(msg["content"])

    # Chat-Eingabe
    if prompt := st.chat_input("Stelle eine Frage zur Kfz-Haftpflichtversicherung..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🎓"):
            # Prüfen ob es eine Begrüßung/Smalltalk ist
            if is_greeting(prompt):
                answer = generate_greeting_response(
                    prompt,
                    st.session_state.chat_history[:-1]
                )
                context_chunks = []
            else:
                with st.spinner("Suche in der Wissensbasis..."):
                    standalone_query = rewrite_question_with_history(
                        st.session_state.chat_history[:-1],
                        prompt
                    )
                    context_chunks = retrieve(standalone_query)

                    if not context_chunks:
                        answer = (
                            "Leider konnte ich keine relevanten Informationen in der "
                            "Wissensbasis finden. Bitte formuliere deine Frage anders "
                            "oder stelle sicher, dass die relevanten Dokumente hochgeladen sind."
                        )
                    else:
                        answer = generate_answer(
                            prompt,
                            context_chunks,
                            st.session_state.chat_history[:-1]
                        )

            st.markdown(answer)

            # Quellen nur anzeigen wenn die Antwort tatsächlich auf Kontext basiert
            answer_lower = answer.lower()
            show_sources = context_chunks and not any(h in answer_lower for h in NO_ANSWER_HINTS)

            if show_sources:
                with st.expander("Verwendete Quellen anzeigen"):
                    for chunk in context_chunks:
                        safe_filename = html.escape(chunk['filename'])
                        safe_content = html.escape(chunk['content'][:250])
                        suffix = "..." if len(chunk['content']) > 250 else ""
                        st.markdown(f"""
                        <div class="source-chunk">
                            <div class="source-header">
                                📄 {safe_filename} — Seite {chunk['page_number']}
                                &nbsp;&middot;&nbsp; Relevanz: {chunk.get('rerank_score', 0):.3f}
                            </div>
                            <div class="source-text">{safe_content}{suffix}</div>
                        </div>
                        """, unsafe_allow_html=True)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        if st.session_state.session_id:
            save_chat_message(st.session_state.session_id, "user", prompt)
            save_chat_message(st.session_state.session_id, "assistant", answer)

# ── Quiz-Modus ──
elif st.session_state.mode == "Quiz":
    st.markdown('<div class="mode-badge mode-badge-quiz">🧠 Quiz-Modus</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="main-header">
        <span class="header-icon">🧠</span>
        <h1>Quiz-Modus</h1>
        <div class="header-sub">Teste dein Wissen zur Kfz-Haftpflichtversicherung</div>
    </div>
    """, unsafe_allow_html=True)

    # Welcome-Karten wenn noch keine Frage generiert
    if not st.session_state.quiz_data and st.session_state.quiz_score["total"] == 0:
        st.markdown("""
        <div class="welcome-grid">
            <div class="welcome-card">
                <span class="card-icon">🎯</span>
                <div class="card-title">Multiple-Choice</div>
                <div class="card-desc">Beantworte Fragen mit vier Antwortmöglichkeiten — generiert aus den Vorlesungsmaterialien.</div>
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

    # Score anzeigen
    score = st.session_state.quiz_score
    if score["total"] > 0:
        pct = score["correct"] / score["total"]
        st.markdown(f"""
        <div class="score-badge">
            🏆 {score['correct']} / {score['total']} richtig ({pct:.0%})
        </div>
        """, unsafe_allow_html=True)
        st.progress(pct)

    # Quiz-Thema
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

    # Enter im Textfeld oder Button-Klick generiert neue Frage
    topic_submitted = quiz_topic and quiz_topic != st.session_state.get("last_quiz_topic", "")
    if generate_btn or topic_submitted:
        if topic_submitted:
            st.session_state.last_quiz_topic = quiz_topic
        with st.spinner("Generiere Frage..."):
            quiz_data = generate_quiz_question(quiz_topic if quiz_topic else None)
            if quiz_data:
                st.session_state.quiz_data = quiz_data
                st.session_state.quiz_answered = False
            else:
                st.error("Konnte keine Frage generieren. Ist die Wissensbasis gefuellt?")

    # Frage anzeigen
    if st.session_state.quiz_data:
        qd = st.session_state.quiz_data

        st.markdown(f"""
        <div class="quiz-question-card">
            <div class="question-text">{qd['question']}</div>
            <div class="question-source">Quelle: {qd.get('source', 'Vorlesungsmaterialien')}</div>
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

# ── Sparring-Modus (Sokratisch) ──
elif st.session_state.mode == "Sparring":

    st.markdown('<div class="mode-badge mode-badge-sparring">⚡ Sparring-Modus</div>', unsafe_allow_html=True)

    # Welcome Screen wenn kein Chat-Verlauf
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

    # Chat-Verlauf anzeigen
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"], avatar="🧑" if msg["role"] == "user" else "⚡"):
            st.markdown(msg["content"])

    # Chat-Eingabe
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
            else:
                with st.spinner("Denke nach..."):
                    standalone_query = rewrite_question_with_history(
                        st.session_state.chat_history[:-1],
                        prompt
                    )
                    context_chunks = retrieve(standalone_query)

                    if not context_chunks:
                        answer = (
                            "Leider konnte ich keine relevanten Informationen in der "
                            "Wissensbasis finden. Bitte formuliere deine Frage anders "
                            "oder stelle sicher, dass die relevanten Dokumente hochgeladen sind."
                        )
                    else:
                        answer = generate_sparring_response(
                            prompt,
                            context_chunks,
                            st.session_state.chat_history[:-1]
                        )

            st.markdown(answer)

            # Quellen im Sparring-Modus anzeigen (gleiche Logik wie Chat)
            answer_lower = answer.lower()
            show_sources = context_chunks and not any(h in answer_lower for h in NO_ANSWER_HINTS)

            if show_sources:
                with st.expander("Verwendete Quellen anzeigen"):
                    for chunk in context_chunks:
                        safe_filename = html.escape(chunk['filename'])
                        safe_content = html.escape(chunk['content'][:250])
                        suffix = "..." if len(chunk['content']) > 250 else ""
                        st.markdown(f"""
                        <div class="source-chunk">
                            <div class="source-header">
                                📄 {safe_filename} — Seite {chunk['page_number']}
                                &nbsp;&middot;&nbsp; Relevanz: {chunk.get('rerank_score', 0):.3f}
                            </div>
                            <div class="source-text">{safe_content}{suffix}</div>
                        </div>
                        """, unsafe_allow_html=True)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        if st.session_state.session_id:
            save_chat_message(st.session_state.session_id, "user", prompt)
            save_chat_message(st.session_state.session_id, "assistant", answer)
