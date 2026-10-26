# tidemark — Docker Desktop (Linux containers)
# Build: docker compose build
# Run:   docker compose run --rm smoke
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TZ=Asia/Taipei \
    TIDEMARK_HOME=/app

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        tzdata \
        curl \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY config/ config/
COPY src/ src/
COPY scripts/ scripts/
COPY tests/ tests/
COPY AGENTS.md CLAUDE.md README.md ./

# Persistent mounts expected: data/, outputs/, .env
RUN mkdir -p data/cache data/raw outputs logs

# Default: smoke (safe). Compose overrides for live/sim.
CMD ["python", "scripts/smoke_check.py"]
