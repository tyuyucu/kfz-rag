# syntax=docker/dockerfile:1.6
#
# image fuer die streamlit-app
# modelle werden erst zur laufzeit ins hf-cache-volume gezogen
# dadurch bleibt das image schlank

FROM python:3.12-slim

# build-tools für psycopg2-binary und sentence-transformers, git für die git+https-dependency
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# requirements separat cachen damit code-aenderungen den layer-cache nicht killen
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# rest vom code
COPY . .

# streamlit an 0.0.0.0 binden damit der host-browser drauf zugreifen kann
EXPOSE 8501

# hf-cache als volume damit modelle ueber docker compose down hinweg bleiben
ENV HF_HOME=/root/.cache/huggingface \
    PYTHONUNBUFFERED=1

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -fsS http://localhost:8501/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
