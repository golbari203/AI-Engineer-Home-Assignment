FROM python:3.12-slim

# poppler-utils provides pdftotext, pdftoppm, pdfinfo; tesseract-ocr provides
# OCR and orientation detection. Without them six documents in this corpus
# degrade or fail outright, including two whole statements' worth of content,
# so they are part of the artefact rather than an assumption about the host.
RUN apt-get update && apt-get install -y --no-install-recommends \
        poppler-utils \
        tesseract-ocr \
        tesseract-ocr-eng \
        tesseract-ocr-osd \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model in, so first run needs no model download.
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('BAAI/bge-small-en-v1.5')"

COPY . .
ENV PYTHONPATH=/app PORTFOLIO_DIR=/data/portfolio
EXPOSE 8501

CMD ["sh", "-c", "python scripts/bootstrap.py && \
     streamlit run app/ui.py --server.port=8501 --server.address=0.0.0.0"]
