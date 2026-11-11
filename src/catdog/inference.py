"""Prédiction en production : décodage sûr de l'image, modèle ONNX, garde-fou et carte d'occlusion.

Ne dépend que de numpy, Pillow et onnxruntime (pas de TensorFlow) : l'API et la démo restent légères.
"""
from __future__ import annotations

import hashlib
import io
import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_BYTES = 8 * 1024 * 1024            # 8 Mo par image
MAX_PIXELS = 40_000_000                # au-delà : refus (bombe de décompression)
MIN_SIDE = 32
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
Image.MAX_IMAGE_PIXELS = MAX_PIXELS


class ImageError(ValueError):
    """Image refusée (format, taille, contenu illisible)."""


def decode(data: bytes) -> Image.Image:
    """Décode une image envoyée par un utilisateur, avec des contrôles stricts.

    Le format est déterminé par le contenu (pas par l'extension ni l'en-tête HTTP). Les métadonnées EXIF ne sont
    utilisées que pour l'orientation, puis abandonnées ; l'image n'est jamais écrite sur disque.
    """
    if not data:
        raise ImageError("Fichier vide.")
    if len(data) > MAX_BYTES:
        raise ImageError(f"Image trop lourde (maximum {MAX_BYTES // (1024 * 1024)} Mo).")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as probe:
                fmt = probe.format
                probe.verify()
            if fmt not in ALLOWED_FORMATS:
                raise ImageError("Format non accepté : JPEG, PNG ou WEBP uniquement.")
            im = Image.open(io.BytesIO(data))
            if im.width * im.height > MAX_PIXELS:
                raise ImageError("Image trop grande.")
            im = ImageOps.exif_transpose(im).convert("RGB")
    except ImageError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError,
            Image.DecompressionBombWarning) as exc:
        raise ImageError("Fichier illisible ou corrompu.") from exc
    if min(im.size) < MIN_SIDE:
        raise ImageError(f"Image trop petite (minimum {MIN_SIDE} pixels de côté).")
    return im


def to_array(im: Image.Image, size: int) -> np.ndarray:
    return np.asarray(im.resize((size, size), Image.BILINEAR), dtype=np.float32)[None]


@dataclass
class Prediction:
    label: str              # "cat", "dog" ou "unsure"
    probability_dog: float
    confidence: float       # probabilité de la classe annoncée
    pet_score: float        # masse ImageNet « chat ou chien » (garde-fou)
    looks_like_pet: bool


class Predictor:
    def __init__(self, model_dir: Path):
        import onnxruntime as ort
        self.meta = json.loads((model_dir / "model.meta.json").read_text())
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        self.session = self._load(ort, model_dir / "model.onnx", self.meta["model_sha256"], opts)
        gate = self.meta["pet_gate"]
        self.gate = self._load(ort, model_dir / gate["file"], gate["sha256"], opts)
        self.size, self.gate_size, self.gate_threshold = self.meta["input_size"], gate["input_size"], gate["threshold"]
        self.band = tuple(self.meta["unsure_band"])

    @staticmethod
    def _load(ort, path: Path, expected_sha: str, opts):
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected_sha:
            raise RuntimeError(f"Empreinte inattendue pour {path.name} : chargement refusé.")
        return ort.InferenceSession(data, sess_options=opts, providers=["CPUExecutionProvider"])

    def proba_dog(self, x: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.session.get_inputs()[0].name: x.astype(np.float32)})[0].ravel()

    def pet_score(self, im: Image.Image) -> float:
        probs = self.gate.run(None, {self.gate.get_inputs()[0].name: to_array(im, self.gate_size)})[0][0]
        return float(probs[151:269].sum() + probs[281:286].sum())

    def predict(self, im: Image.Image) -> Prediction:
        p = float(self.proba_dog(to_array(im, self.size))[0])
        lo, hi = self.band
        label = "dog" if p >= hi else "cat" if p <= lo else "unsure"
        score = self.pet_score(im)
        return Prediction(label=label, probability_dog=p, confidence=max(p, 1 - p), pet_score=score,
                          looks_like_pet=score >= self.gate_threshold)

    def occlusion(self, im: Image.Image, grid: int = 8) -> np.ndarray:
        """Carte d'occlusion : on masque tour à tour chaque case d'une grille et on mesure de combien la
        probabilité de la classe prédite baisse. Plus la case est chaude, plus elle a compté pour la décision."""
        x = to_array(im, self.size)[0]
        p0 = float(self.proba_dog(x[None])[0])
        sign = 1.0 if p0 >= 0.5 else -1.0
        step = int(np.ceil(self.size / grid))
        grey = x.mean(axis=(0, 1))
        batch = []
        for i in range(grid):
            for j in range(grid):
                xm = x.copy()
                xm[i * step:(i + 1) * step, j * step:(j + 1) * step] = grey
                batch.append(xm)
        p = self.proba_dog(np.stack(batch))
        drop = sign * (p0 - p)               # baisse de confiance dans la classe prédite
        return np.clip(drop, 0, None).reshape(grid, grid)
