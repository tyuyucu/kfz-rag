# kfz-rag — Lernassistent „Kfz-Haftpflichtversicherung"

Lokal laufendes RAG-System für die Vorlesung „Kfz-Haftpflicht­versicherung".
Drei Modi (Chat, Quiz, Sparring) über eine Streamlit-Oberfläche, alles
auf Deutsch.

---

## Voraussetzungen

- **Docker Desktop** ([docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop))
  installiert und gestartet — alle anderen Komponenten laufen in
  Containern.
- **OpenAI-API-Schlüssel** für die Embeddings (Kosten: einmalig
  unter 0,01 € für das Ingesten der Vorlesungsmaterialien, danach
  Bruchteile davon pro Suchanfrage).
- ~3 GB freier Festplattenplatz (Postgres + App-Image).
- Mindestens 4 GB RAM für die Container; 8 GB Gesamt-RAM sind
  komfortabel.

## Installation

```bash
# 1) Repository klonen
git clone <repository-url>
cd kfz-rag

# 2) Container bauen und starten
docker compose up -d

# 3) Browser öffnen
# → http://localhost:8501
```

Beim ersten Start öffnet die App einen **Erstkonfigurations-Wizard**.
Folge den Schritten:

1. **OpenAI-Schlüssel hinterlegen.** Den Schlüssel erstellst du auf
   [platform.openai.com/api-keys](https://platform.openai.com/api-keys).
   Er wird ausschließlich lokal in deiner Postgres-DB gespeichert
   und ausschließlich für das Erzeugen von Embeddings verwendet.
   Der Wizard verifiziert den Schlüssel mit einem Mini-Aufruf, bevor
   er gespeichert wird.
2. **PDFs hochladen.** Lade die Lernmaterialien hoch, mit denen du
   arbeiten möchtest (Vorlesungsskript, AKB, weitere Quellen). Du
   kannst diesen Schritt überspringen und PDFs später über die Sidebar
   nachladen.
3. **Fertig.** Beim nächsten Start landest du direkt in der App.

## Chat-Anbieter wählen

Unabhängig vom Embedding entscheidest du in der Sidebar, wie die
Chat-Antworten erzeugt werden:

| Pfad | Schlüssel nötig | Kosten |
|---|---|---|
| OpenAI | eigener Schlüssel | je nach Nutzung |
| Groq (Llama-3.x) | eigener Schlüssel | **kostenlos im Free Tier** |
| Google Gemini | eigener Schlüssel | je nach Nutzung |
| Ollama (lokal) | nicht nötig | kostenlos, läuft offline |

Für den Ollama-Pfad muss **Ollama auf deiner Host-Maschine installiert
und gestartet sein** ([ollama.com](https://ollama.com)). Das Default-
Modell `llama3.2:3b` (~2 GB) wird beim ersten Aufruf automatisch geladen.

## Wissensbasis zurücksetzen

In der Sidebar gibt es unter „⚙ Erweitert" einen Reset-Button. Damit
werden alle Chunks, Dokumente und die Setup-Konfiguration gelöscht —
beim nächsten Start läuft der Wizard erneut. Nützlich, wenn du den
hinterlegten OpenAI-Schlüssel ändern oder die Wissensbasis komplett
neu aufbauen willst.

## Stack

- Python 3.12, Streamlit
- PostgreSQL + pgvector (Vektor- und Volltext-Index)
- Embedding: OpenAI `text-embedding-3-small`
- Chat: OpenAI / Groq / Gemini via `llm_client` oder lokales Ollama
- Cross-Encoder `BAAI/bge-reranker-v2-m3` für Reranking

## Projektstruktur

```
.
├── app.py                  # Streamlit-UI mit Setup-Wizard und 3 Modi
├── config.py               # zentrale Pipeline-Parameter
├── docker-compose.yml      # App + Postgres + Volumes
├── Dockerfile              # Streamlit-App-Image
├── ingestion/              # PDF → Chunk → Embedding → DB
├── retrieval/              # Multi-Query, Hybrid-Search, RRF, Reranker
├── generation/             # Antwort-Generator, Quiz, Sparring, Chat-Adapter
├── db/                     # Postgres + pgvector + app_config-Tabelle
└── documents/              # Quell-PDFs (gitignored, Urheberrecht)
```

## Troubleshooting

**Postgres startet nicht / Port 5432 belegt.**
Auf vielen Systemen läuft schon ein Postgres-Dienst auf 5432. Stoppe ihn
oder ändere das Port-Mapping in `docker-compose.yml`
(`"5433:5432"` und in `DATABASE_URL` Port `5433` setzen).

**Ollama wird vom App-Container nicht erreicht (Linux).**
`host.docker.internal` ist auf Linux nicht automatisch verfügbar.
Aktiviere im `app:`-Service der `docker-compose.yml`:
```yaml
extra_hosts:
  - "host.docker.internal:host-gateway"
```
Anschließend `docker compose up -d --force-recreate app`.

**OpenAI-Schlüssel wird im Wizard abgelehnt.**
Der Wizard macht einen Mini-Embed-Aufruf zur Validierung. Schlägt das
fehl, ist der Schlüssel ungültig oder dem Konto fehlt das Embeddings-
Kontingent. Schlüssel auf [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
prüfen.
