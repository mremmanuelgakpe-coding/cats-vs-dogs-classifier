"""Transfert learning, étape 1 : extraction des caractéristiques des réseaux pré-entraînés (gelés).

Chaque image passe une fois dans VGG16, ResNet50, InceptionV3 et MobileNetV2 (poids ImageNet, sans la tête de
classification, moyenne globale en sortie). Les vecteurs obtenus sont mis en cache : la recherche
d'hyperparamètres de la tête de classification se fait ensuite en quelques secondes au lieu de plusieurs heures.

Usage : PYTHONPATH=src python -m catdog.features [vgg16 resnet50 ...]
"""
from __future__ import annotations

import hashlib
import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402

from . import config  # noqa: E402
from .data import load_manifest  # noqa: E402


def weights_path(name: str):
    """Chemin des poids ImageNet, après contrôle de leur empreinte SHA-256."""
    url, sha = config.WEIGHTS[name]
    path = config.WEIGHTS_DIR / url.rsplit("/", 1)[1]
    if not path.exists():
        raise FileNotFoundError(f"Poids absents : {path}. Télécharger : make weights")
    if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
        raise ValueError(f"Empreinte SHA-256 inattendue pour {path.name}")
    return path


def backbone(name: str, size: int, pooling="avg"):
    import keras
    A = keras.applications
    ctor = {"vgg16": A.VGG16, "resnet50": A.ResNet50, "inception_v3": A.InceptionV3, "mobilenet_v2": A.MobileNetV2}[name]
    return ctor(include_top=False, weights=str(weights_path(name)), pooling=pooling, input_shape=(size, size, 3))


def preprocess(name: str):
    import keras
    A = keras.applications
    return {"vgg16": A.vgg16.preprocess_input, "resnet50": A.resnet50.preprocess_input,
            "inception_v3": A.inception_v3.preprocess_input, "mobilenet_v2": A.mobilenet_v2.preprocess_input}[name]


def load_images(paths, size: int) -> np.ndarray:
    """Images RVB redimensionnées (size × size), float32 dans [0, 255]."""
    from PIL import Image
    out = np.empty((len(paths), size, size, 3), dtype=np.float32)
    for i, rel in enumerate(paths):
        with Image.open(config.DATA_SRC / rel) as im:
            out[i] = np.asarray(im.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.float32)
    return out


def extract(name: str, chunk: int = 512) -> None:
    size = config.BACKBONES[name]
    model, pre = backbone(name, size), preprocess(name)
    m = load_manifest()
    feats = []
    t0 = time.time()
    for start in range(0, len(m), chunk):
        x = pre(load_images(m["path"].iloc[start:start + chunk], size))
        feats.append(model.predict(x, batch_size=32, verbose=0).astype(np.float16))
        done = start + len(x)
        print(f"{name}: {done}/{len(m)} images, {done / (time.time() - t0):.1f} img/s", flush=True)
    config.FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(config.FEATURES_DIR / f"{name}.npz", X=np.concatenate(feats), path=m["path"].to_numpy(),
                        y=m["y"].to_numpy(), split=m["split"].to_numpy(), seconds=time.time() - t0)


if __name__ == "__main__":
    for n in sys.argv[1:] or list(config.BACKBONES):
        extract(n)
