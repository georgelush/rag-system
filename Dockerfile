FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN adduser --disabled-password --gecos "" raguser

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY rag_server.py .
COPY rag/ rag/

RUN chown -R raguser:raguser /app
USER raguser

EXPOSE 8080

CMD ["uvicorn", "rag_server:app", "--host", "0.0.0.0", "--port", "8080"]
