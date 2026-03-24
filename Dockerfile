FROM python:3.11-slim

LABEL org.opencontainers.image.title="SentinelAI"
LABEL org.opencontainers.image.description="AI Reliability & Observability System"
LABEL org.opencontainers.image.source="https://github.com/sentinel-ai/sentinel-ai"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY sentinel/ ./sentinel/
COPY dashboard/ ./dashboard/

# Non-root user for security
RUN useradd -m -u 1000 sentinel && chown -R sentinel:sentinel /app
USER sentinel

EXPOSE 8765

HEALTHCHECK --interval=10s --timeout=5s --retries=3 \
  CMD curl -f http://localhost:8765/health || exit 1

CMD ["uvicorn", "sentinel.server:app", \
     "--host", "0.0.0.0", \
     "--port", "8765", \
     "--log-level", "info", \
     "--access-log"]
