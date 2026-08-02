FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ORT_NUM_THREADS=2 \
    INSIGHTFACE_HOME=/app/data/models

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY scripts ./scripts

# Precarga el modelo buffalo_sc para que el contenedor no dependa de red al iniciar
RUN python -c "\
import os; os.makedirs('/app/data/models', exist_ok=True); \
from insightface.app import FaceAnalysis; \
fa = FaceAnalysis(name='buffalo_sc', root='/app/data/models'); \
fa.prepare(ctx_id=-1)"

RUN mkdir -p /app/data/media

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
