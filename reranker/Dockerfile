FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
# sentence-transformers pulls torch — CPU-only wheel is sufficient and much smaller
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download the model at build time so the first request isn't slow
RUN python -c \
    "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY app.py .

RUN useradd -m -u 1001 reranker
USER reranker

EXPOSE 8090
CMD ["python", "app.py"]
