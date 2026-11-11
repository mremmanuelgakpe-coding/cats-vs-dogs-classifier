"""Réseaux entraînés sur les images : CNN maison (de zéro) et MobileNetV2 affiné (fine-tuning).

- CNN maison : 4 blocs convolution + normalisation + max-pooling (12 époques au plus), augmentation (miroir, rotation, zoom), 128 px.
- MobileNetV2 affiné : 1) tête entraînée sur le réseau gelé, 2) dégel des derniers blocs avec un taux
  d'apprentissage 50 fois plus faible. Le prétraitement est intégré au modèle : il reçoit des pixels 0-255.

Usage : PYTHONPATH=src python -m catdog.nets cnn | finetune
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402

from . import config  # noqa: E402
from .data import load_manifest  # noqa: E402
from .features import load_images, weights_path  # noqa: E402

CACHE = config.ROOT / "data" / "cache"


def images(size: int):
    """Images de l'échantillon en uint8 (mises en cache sur disque, jamais publiées)."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"images_{size}.npz"
    m = load_manifest()
    if not f.exists():
        X = np.concatenate([load_images(m["path"].iloc[i:i + 1000], size).astype(np.uint8)
                            for i in range(0, len(m), 1000)])
        np.savez(f, X=X)
    X = np.load(f)["X"]
    return {s: (X[(m["split"] == s).to_numpy()], m.loc[m["split"] == s, "y"].to_numpy()) for s in ("train", "val", "test")}


def augmentation():
    import keras
    return keras.Sequential([keras.layers.RandomFlip("horizontal"), keras.layers.RandomRotation(0.08),
                             keras.layers.RandomZoom(0.12), keras.layers.RandomContrast(0.1)], name="augmentation")


def build_cnn(size=config.CNN_SIZE):
    import keras
    L = keras.layers
    inp = keras.Input((size, size, 3))
    x = L.Rescaling(1 / 255)(augmentation()(inp))
    for filters in (32, 64, 128, 256):      # une convolution par bloc : budget de calcul CPU
        x = L.Conv2D(filters, 3, padding="same", use_bias=False)(x)
        x = L.BatchNormalization()(x)
        x = L.Activation("relu")(x)
        x = L.MaxPooling2D()(x)
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dropout(0.4)(x)
    out = L.Dense(1, activation="sigmoid")(x)
    return keras.Model(inp, out, name="cnn_maison")


def build_finetune(size=config.FT_SIZE):
    import keras
    inp = keras.Input((size, size, 3), name="image")
    x = augmentation()(inp)
    x = keras.applications.mobilenet_v2.preprocess_input(x)
    base = keras.applications.MobileNetV2(include_top=False, weights=str(weights_path("mobilenet_v2")),
                                          input_shape=(size, size, 3), pooling="avg")
    base.trainable = False
    x = base(x, training=False)       # normalisation par lots en mode inférence, même pendant le dégel
    x = keras.layers.Dropout(0.3)(x)
    out = keras.layers.Dense(1, activation="sigmoid", name="proba_chien")(x)
    return keras.Model(inp, out, name="mobilenet_v2_ft"), base


def callbacks(patience=4):
    import keras
    return [keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6)]


def history_dict(*hists):
    out = {}
    for h in hists:
        for k, v in h.history.items():
            out.setdefault(k, []).extend(float(x) for x in v)
    return out


def train_cnn():
    import keras
    keras.utils.set_random_seed(config.SEED)
    d = images(config.CNN_SIZE)
    m = build_cnn()
    m.compile(optimizer=keras.optimizers.AdamW(1e-3, weight_decay=1e-4), loss="binary_crossentropy", metrics=["accuracy"])
    t0 = time.time()
    h = m.fit(*d["train"], validation_data=d["val"], epochs=12, batch_size=64, callbacks=callbacks(4), verbose=2)
    save(m, "cnn_scratch", d, history_dict(h), time.time() - t0)


def train_finetune(unfreeze_from=100):
    import keras
    keras.utils.set_random_seed(config.SEED)
    d = images(config.FT_SIZE)
    m, base = build_finetune()
    m.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    t0 = time.time()
    h1 = m.fit(*d["train"], validation_data=d["val"], epochs=4, batch_size=64, callbacks=callbacks(2), verbose=2)
    base.trainable = True
    for layer in base.layers[:unfreeze_from]:
        layer.trainable = False
    m.compile(optimizer=keras.optimizers.Adam(2e-5), loss="binary_crossentropy", metrics=["accuracy"])
    h2 = m.fit(*d["train"], validation_data=d["val"], epochs=8, batch_size=64, callbacks=callbacks(3), verbose=2)
    info = {"unfreeze_from_layer": unfreeze_from, "base_layers": len(base.layers),
            "trainable_params": int(sum(np.prod(w.shape) for w in m.trainable_weights)),
            "stage1_epochs": len(h1.history["loss"]), "stage2_epochs": len(h2.history["loss"])}
    save(m, "mobilenet_v2_ft", d, history_dict(h1, h2), time.time() - t0, info)


def save(m, name, d, hist, seconds, extra=None):
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORTS_DIR / "predictions").mkdir(parents=True, exist_ok=True)
    m.save(config.ROOT / "data" / "cache" / f"{name}.keras")
    np.save(config.REPORTS_DIR / "predictions" / f"{name}_test.npy", m.predict(d["test"][0], batch_size=64, verbose=0).ravel())
    np.save(config.REPORTS_DIR / "predictions" / f"{name}_val.npy", m.predict(d["val"][0], batch_size=64, verbose=0).ravel())
    (config.REPORTS_DIR / f"history_{name}.json").write_text(json.dumps(
        {"history": hist, "train_seconds": seconds, "params": int(m.count_params()), **(extra or {})}, indent=2))


if __name__ == "__main__":
    {"cnn": train_cnn, "finetune": train_finetune}[sys.argv[1]]()
