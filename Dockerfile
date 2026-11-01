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

# Bake the embedding model in, so first run needs no model download. It lives
# outside /root so a non-root runtime user can read it too.
ENV HF_HOME=/opt/hf-cache
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('BAAI/bge-small-en-v1.5')" \
    && chmod -R a+rwX /opt/hf-cache

COPY . .
# Some hosts run the container as uid 1000 rather than root. Giving
# that uid a home holds the extraction cache, and an open portfolio folder
# lets uploads and the watcher write. Root (docker-compose, Render) is unaffected.
RUN useradd -m -u 1000 user && chmod -R a+rwX /app/portfolio
# docker-compose mounts the live folder at /data/portfolio and says so; on a
# host without that mount (Render) the copy baked into the image is read.
ENV PYTHONPATH=/app PORTFOLIO_DIR=/app/portfolio
EXPOSE 8501

# MySQL is loaded only where one is configured (docker-compose sets
# MYSQL_HOST); the app itself reads the portfolio folder. Hosts that assign
# a port pass it in PORT.
CMD ["sh", "-c", "{ [ -z \"$MYSQL_HOST\" ] || python scripts/bootstrap.py; } && \
     streamlit run app/ui.py --server.port=${PORT:-8501} --server.address=0.0.0.0 \
     --server.headless=true"]
