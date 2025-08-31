# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Optional build deps (kept minimal). Remove if wheels are sufficient.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files needed for install
COPY pyproject.toml README.md LICENSE MANIFEST.in ./
COPY synthetic_data_kit ./synthetic_data_kit
COPY configs ./configs

# Install package
RUN pip install --upgrade pip \
    && pip install .

# Default data directory inside container (can be overridden in compose)
RUN mkdir -p /app/data
ENV SDK_BROWSE_ROOT=/app/data

EXPOSE 5000

# Start Flask server without changing default port (5000) and bind to 0.0.0.0
CMD ["python", "-c", "from synthetic_data_kit.server.app import run_server; run_server(host='0.0.0.0', port=5000, debug=False)"]
