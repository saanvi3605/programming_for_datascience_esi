# ─────────────────────────────────────────────────────────────────────────────
#  Economic Stress Index — Dockerfile
#  Builds a single image that can run EITHER:
#    • the data pipeline  →  docker compose run pipeline
#    • the Streamlit app  →  docker compose up app
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.12-slim

# ── System deps ───────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies (cached layer) ───────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# ── Copy project source ───────────────────────────────────────────────────────
COPY . .

# ── Create output directories (persist via volumes) ───────────────────────────
RUN mkdir -p data/raw data/processed results/figures results/tables notebooks

# ── Streamlit config (no browser popup, listen on all interfaces) ─────────────
RUN mkdir -p /root/.streamlit
COPY docker/streamlit_config.toml /root/.streamlit/config.toml

# ── Default command: launch the dashboard ─────────────────────────────────────
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
