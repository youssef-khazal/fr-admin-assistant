# The web app (search + chat page). The LLM runs in a separate Ollama container (see docker-compose.yml).
# The search index (data/processed, data/index) is built beforehand with scripts/build_corpus.py and
# scripts/embed.py and mounted as a volume; the search models are downloaded on first start.
FROM python:3.11-slim

WORKDIR /app
COPY pyproject.toml requirements.txt README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[app]"

ENV DATA_DIR=/app/data OLLAMA_URL=http://ollama:11434 THINK=1
EXPOSE 8000
CMD ["uvicorn", "fr_admin_assistant.app:app", "--host", "0.0.0.0", "--port", "8000"]
