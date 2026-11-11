"""Évaluation finale sur le test, choix du modèle déployé, figures et export ONNX.

Règle de choix (définie avant de regarder le test) : parmi les modèles légers (< 20 Mo), celui qui a la plus
petite perte logarithmique en **validation**. Les gros réseaux (VGG16, ResNet50, InceptionV3) servent de
référence : s'ils faisaient nettement mieux, on le verrait sur le test.

Usage : PYTHONPATH=src python -m catdog.evaluate
"""
from __future__ import annotations

import hashlib
import json
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix, f1_score, log_loss,  # noqa: E402
                             roc_auc_score)

from . import config  # noqa: E402
from .data import load_manifest  # noqa: E402

PRED = config.REPORTS_DIR / "predictions"
CANDIDATES = ["cnn_scratch", "vgg16", "resnet50", "inception_v3", "mobilenet_v2", "mobilenet_v2_ft"]
LIGHT = {"mobilenet_v2", "mobilenet_v2_ft", "cnn_scratch"}
# Classes ImageNet utilisées par le garde-fou « est-ce bien un chat ou un chien ? »
IMAGENET_DOGS, IMAGENET_CATS = range(151, 269), range(281, 286)


def metrics(y, p) -> dict:
    p = np.clip(p, 1e-7, 1 - 1e-7)
    yhat = (p >= 0.5).astype(int)
    rng = np.random.default_rng(config.SEED)
    boot = [accuracy_score(y[i], yhat[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(1000))]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    lo_b, hi_b = config.UNSURE_BAND
    sure = (p <= lo_b) | (p >= hi_b)
    return {"accuracy": accuracy_score(y, yhat), "accuracy_ci95": [float(lo), float(hi)], "roc_auc": roc_auc_score(y, p),
            "log_loss": log_loss(y, p), "brier": brier_score_loss(y, p), "f1": f1_score(y, yhat),
            "errors": int((yhat != y).sum()), "confusion": confusion_matrix(y, yhat).tolist(),
            "confident_share": float(sure.mean()), "confident_accuracy": float((yhat[sure] == y[sure]).mean()),
            "unsure_count": int((~sure).sum())}


def val_loss(name, search):
    if name in search:
        r = search[name]
        return min(r["random_search_best"]["val_loss"], r["logreg_grid"]["val_loss"])
    y = load_manifest("val")["y"].to_numpy()
    return log_loss(y, np.clip(np.load(PRED / f"{name}_val.npy"), 1e-7, 1 - 1e-7))


def export_deployed(name: str):
    """Exporte le modèle choisi en ONNX (prétraitement inclus) et renvoie (chemin, taille d'entrée)."""
    import keras
    if name == "mobilenet_v2_ft":
        m = keras.models.load_model(config.ROOT / "data" / "cache" / "mobilenet_v2_ft.keras")
        size = config.FT_SIZE
    elif name in config.BACKBONES:     # réseau pré-entraîné gelé + tête retenue par la recherche aléatoire
        from .features import backbone, preprocess
        size = config.BACKBONES[name]
        head = keras.models.load_model(config.ROOT / "data" / "cache" / f"head_{name}.keras")
        inp = keras.Input((size, size, 3), name="image")
        m = keras.Model(inp, head(backbone(name, size)(preprocess(name)(inp))), name=f"{name}_tl")
    else:
        raise ValueError(f"export non prévu pour {name}")
    m.predict(np.zeros((1, size, size, 3), np.float32), verbose=0)
    path = config.MODELS_DIR / "model.onnx"
    m.export(str(path), format="onnx", verbose=False)
    return path, size


def export_pet_gate():
    """MobileNetV2 complet (1 000 classes ImageNet) : sert à signaler les photos sans chat ni chien."""
    import keras
    from .features import weights_path
    w = weights_path("mobilenet_v2_top")
    inp = keras.Input((224, 224, 3), name="image")
    net = keras.applications.MobileNetV2(include_top=True, weights=str(w), input_shape=(224, 224, 3))
    m = keras.Model(inp, net(keras.applications.mobilenet_v2.preprocess_input(inp)))
    m.predict(np.zeros((1, 224, 224, 3), np.float32), verbose=0)
    path = config.MODELS_DIR / "pet_gate.onnx"
    m.export(str(path), format="onnx", verbose=False)
    return path


def pet_score(probs: np.ndarray) -> np.ndarray:
    return probs[:, list(IMAGENET_DOGS)].sum(1) + probs[:, list(IMAGENET_CATS)].sum(1)


def onnx_latency(path, size, n=50):
    import onnxruntime as ort
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    x = (np.random.default_rng(0).random((1, size, size, 3)) * 255).astype(np.float32)
    name = s.get_inputs()[0].name
    for _ in range(5):
        s.run(None, {name: x})
    t = time.perf_counter()
    for _ in range(n):
        s.run(None, {name: x})
    return 1000 * (time.perf_counter() - t) / n, s


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    search = json.loads((config.REPORTS_DIR / "transfer_search.json").read_text())
    test = load_manifest("test")
    y = test["y"].to_numpy()
    rows, probs = [], {}
    for name in CANDIDATES:
        f = PRED / f"{name}_test.npy"
        if not f.exists():
            continue
        p = np.load(f)
        probs[name] = p
        r = {"model": name, "label": config.BACKBONE_LABELS[name], "val_log_loss": val_loss(name, search), **metrics(y, p)}
        if name in search:
            r["approach"] = "Réseau pré-entraîné gelé + tête " + ("Keras" if search[name]["selected_head"] == "keras" else "logistique")
            r["input_size"] = search[name]["input_size"]
            r["extract_ms_per_image"] = search[name]["extract_ms_per_image"]
        else:
            h = json.loads((config.REPORTS_DIR / f"history_{name}.json").read_text())
            r["approach"] = "Entraîné de zéro" if name == "cnn_scratch" else "Pré-entraîné puis affiné (fine-tuning)"
            r["input_size"] = config.CNN_SIZE if name == "cnn_scratch" else config.FT_SIZE
            r["train_minutes"] = h["train_seconds"] / 60
            r["params"] = h["params"]
        rows.append(r)
    table = pd.DataFrame(rows)
    deployed = table[table["model"].isin(LIGHT)].sort_values("val_log_loss").iloc[0]["model"]
    table["deployed"] = table["model"] == deployed
    table.drop(columns=["confusion"]).to_csv(config.REPORTS_DIR / "model_comparison.csv", index=False)

    # Export ONNX, contrôle d'équivalence Keras / ONNX et latence
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path, size = export_deployed(deployed)
    ms, sess = onnx_latency(path, size)
    from .features import load_images
    xs = load_images(test["path"].iloc[:64], size)
    onnx_p = np.concatenate([sess.run(None, {sess.get_inputs()[0].name: xs[i:i + 1]})[0].ravel() for i in range(len(xs))])
    max_gap = float(np.abs(onnx_p - probs[deployed][:64]).max())

    gate_path = export_pet_gate()
    gate_ms, gate = onnx_latency(gate_path, 224)
    val = load_manifest("val")
    xv = load_images(val["path"], 224)
    gate_probs = np.concatenate([gate.run(None, {gate.get_inputs()[0].name: xv[i:i + 32]})[0] for i in range(0, len(xv), 32)])
    scores = pet_score(gate_probs)
    gate_threshold = float(np.quantile(scores, 0.01))   # 99 % des vraies photos de chats/chiens passent

    best = table.set_index("model").loc[deployed]
    meta = {"name": "cats-vs-dogs", "version": "1.0.0", "model": deployed, "label": config.BACKBONE_LABELS[deployed],
            "input_size": size, "input": "image RVB, pixels 0-255, float32, NHWC ; prétraitement inclus dans le graphe",
            "output": "probabilité que l'image soit un chien", "classes": config.CLASSES, "unsure_band": config.UNSURE_BAND,
            "model_sha256": sha(path), "model_mb": round(path.stat().st_size / 1e6, 1), "latency_ms_cpu": round(ms, 1),
            "keras_onnx_max_gap": max_gap,
            "pet_gate": {"file": gate_path.name, "sha256": sha(gate_path), "input_size": 224, "threshold": gate_threshold,
                         "latency_ms_cpu": round(gate_ms, 1), "mb": round(gate_path.stat().st_size / 1e6, 1),
                         "val_pet_score_median": float(np.median(scores))},
            "test": {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                     for k, v in best.drop(labels=["label", "approach"]).to_dict().items()
                     if k not in ("deployed",)},
            "data_commit": config.DATA_COMMIT, "trained_on": "8 000 photos (4 000 chats, 4 000 chiens)"}
    (config.MODELS_DIR / "model.meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=float))
    out = {"deployed": deployed, "selection_rule": "modèle léger (< 20 Mo) de plus faible perte log en validation",
           "test_size": int(len(y)), "models": table.drop(columns=["deployed"]).to_dict(orient="records"),
           "deployed_confusion": best["confusion"], "onnx": {"latency_ms": ms, "max_gap": max_gap},
           "pet_gate_threshold": gate_threshold}
    (config.REPORTS_DIR / "metrics.json").write_text(json.dumps(out, indent=2, ensure_ascii=False, default=float))
    # Erreurs du modèle déployé : identifiants seulement (les images ne sont pas publiées)
    p = probs[deployed]
    err = test.assign(p_dog=p)[((p >= 0.5).astype(int) != y)][["path", "label", "p_dog"]]
    err.to_csv(config.REPORTS_DIR / "deployed_errors.csv", index=False)
    print(table[["label", "accuracy", "roc_auc", "log_loss", "val_log_loss", "errors"]].to_string())
    print("déployé :", deployed, f"| ONNX {ms:.1f} ms, écart max {max_gap:.2e} | garde-fou seuil {gate_threshold:.3f}")


if __name__ == "__main__":
    main()
