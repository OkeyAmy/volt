FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies from pyproject.toml
COPY pyproject.toml .
RUN pip install --no-cache-dir --default-timeout=120 .

# Download NLTK data (VADER lexicon) at build time
RUN python3 -c "import nltk; nltk.download('vader_lexicon', quiet=True)"

# Copy application code, trained artifacts, and product catalog
COPY app/ app/
COPY artifacts/ artifacts/
COPY data/ data/

EXPOSE 8000

# Health-check every 30 s once the container is running
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
