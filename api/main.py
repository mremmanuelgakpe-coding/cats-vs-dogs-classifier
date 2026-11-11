"""API de classification d'images Chat / Chien (FastAPI + ONNX Runtime).

Sécurité :
- clé API (en-tête X-API-Key, comparaison à temps constant) ; documentation désactivée par défaut ;
- image lue en mémoire avec une limite stricte (8 Mo), format reconnu par son contenu (JPEG, PNG, WEBP),
  refus des « bombes de décompression », jamais écrite sur disque ni journalisée ;
- modèles ONNX chargés seulement si leur empreinte SHA-256 correspond aux métadonnées (pas de pickle) ;
- nombre de prédictions simultanées plafonné (protection contre l'épuisement CPU / mémoire).
"""
from __future__ import annotations

import os
import secrets
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, Security, UploadFile, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field

from catdog.inference import MAX_BYTES, ImageError, Predictor, decode

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT / "models"))
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT", "4"))
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_slots = threading.BoundedSemaphore(MAX_CONCURRENT)


class Prediction(BaseModel):
    label: Literal["cat", "dog", "unsure"] = Field(description="« unsure » si la probabilité est entre les bornes")
    label_fr: Literal["Chat", "Chien", "Incertain"]
    probability_dog: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0.5, le=1)
    looks_like_pet: bool = Field(description="False : l'image ne ressemble ni à un chat ni à un chien")
    pet_score: float
    occlusion: list[list[float]] | None = Field(None, description="Grille 8 × 8 : importance de chaque zone")
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = Predictor(MODEL_DIR)
    yield


docs_on = os.getenv("ENABLE_DOCS", "false").lower() == "true"
app = FastAPI(title="Cats vs Dogs API", version="1.0.0", lifespan=lifespan, docs_url="/docs" if docs_on else None,
              redoc_url=None, openapi_url="/openapi.json" if docs_on else None)


def require_api_key(key: str | None = Security(api_key_header)) -> None:
    expected = os.getenv("API_KEY")
    if not expected:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "API_KEY non configurée côté serveur.")
    if not key or not secrets.compare_digest(key, expected):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Clé API invalide.")


@app.middleware("http")
async def security_headers(request: Request, call_next):
    length = request.headers.get("content-length")
    if length and length.isdigit() and int(length) > MAX_BYTES + 64 * 1024:
        from fastapi.responses import JSONResponse
        return JSONResponse({"detail": "Requête trop volumineuse."}, status_code=413)
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Cache-Control"] = "no-store"
    response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
    return response


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": hasattr(app.state, "predictor")}


@app.get("/model-info", dependencies=[Depends(require_api_key)])
def model_info():
    m = app.state.predictor.meta
    return {"model": m["label"], "version": m["version"], "input_size": m["input_size"], "unsure_band": m["unsure_band"],
            "test_accuracy": m["test"]["accuracy"], "test_roc_auc": m["test"]["roc_auc"],
            "latency_ms_cpu": m["latency_ms_cpu"], "model_sha256": m["model_sha256"]}


@app.post("/predict", response_model=Prediction, dependencies=[Depends(require_api_key)])
def predict(image: UploadFile = File(..., description="Photo JPEG, PNG ou WEBP, 8 Mo maximum"),
            explain: bool = Query(False, description="Ajouter la carte d'occlusion 8 × 8 (≈ 65 passages du modèle)")):
    data = image.file.read(MAX_BYTES + 1)
    try:
        im = decode(data)
    except ImageError as exc:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE if "Format" in str(exc) else
                            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE if "lourde" in str(exc) else
                            status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from None
    if not _slots.acquire(timeout=10):
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Serveur occupé, réessayez.")
    try:
        p = app.state.predictor
        r = p.predict(im)
        occ = p.occlusion(im).round(4).tolist() if explain else None
    finally:
        _slots.release()
    meta = p.meta
    return Prediction(label=r.label, label_fr={"cat": "Chat", "dog": "Chien", "unsure": "Incertain"}[r.label],
                      probability_dog=round(r.probability_dog, 4), confidence=round(r.confidence, 4),
                      looks_like_pet=r.looks_like_pet, pet_score=round(r.pet_score, 4), occlusion=occ,
                      model_version=f"{meta['model']}-{meta['version']}")
