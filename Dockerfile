# syntax=docker/dockerfile:1.6
#
# Streamlit-App für den lokalen Studi-Distribution-Pfad.
# Image enthält den Embedding-Stack (sentence-transformers + einops) sowie
# alle Pipeline-Dependencies. Modelle werden zur Laufzeit beim ersten
# Embed-Call ins HuggingFace-Cache-Volume gezogen — dadurch bleibt das
# Image schlank (~2 GB) und Studis ohne Qwen2-Pfad sparen den 3-GB-Pull.

FROM python:3.12-slim

# Build-Tools für psycopg2-binary, einops, sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python-Dependencies separat cachen, damit Code-Änderungen nicht den
# kompletten Layer-Cache invalidieren.
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Restlicher Code
COPY . .

# Streamlit auf 0.0.0.0 binden, damit der Host-Browser drauf zugreifen kann.
EXPOSE 8501

# HuggingFace-Cache an die übliche Stelle, wird per Volume gemountet —
# damit Modelle nach `docker compose down` nicht weg sind.
ENV HF_HOME=/root/.cache/huggingface \
    PYTHONUNBUFFERED=1

# Gesundheits-Check: Streamlit-Default-Endpoint.
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
