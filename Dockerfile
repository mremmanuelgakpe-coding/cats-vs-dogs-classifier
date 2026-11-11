# ---- Image de production de l'API Chat / Chien ----
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src \
    OMP_NUM_THREADS=2

WORKDIR /app
COPY requirements-api.txt .
RUN pip install -r requirements-api.txt

# Code d'inférence et modèles ONNX uniquement (ni données, ni TensorFlow, ni poids d'entraînement)
COPY api/ api/
COPY src/catdog/__init__.py src/catdog/inference.py src/catdog/
COPY models/model.onnx models/pet_gate.onnx models/model.meta.json models/

# Utilisateur non-root, fichiers en lecture seule
RUN useradd --create-home --uid 10001 appuser && chown -R root:root /app && chmod -R a-w /app
USER 10001

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-server-header"]
