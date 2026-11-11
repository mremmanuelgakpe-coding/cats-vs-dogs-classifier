"""Transfert learning, étape 2 : tête de classification sur les caractéristiques gelées.

Pour chaque réseau pré-entraîné :
- **grid search** d'une régression logistique (C) en validation croisée 5 plis sur l'échantillon d'entraînement ;
- **random search** (16 essais) d'une tête Keras : couche cachée (0, 128, 256, 512), dropout, pénalité L2, taux
  d'apprentissage. Arrêt précoce sur la perte de validation.
La meilleure tête (perte de validation la plus basse) est retenue ; le test n'est utilisé qu'une fois, à la fin.
"""
from __future__ import annotations

import json
import os
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import log_loss  # noqa: E402
from sklearn.model_selection import GridSearchCV, StratifiedKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from . import config  # noqa: E402

SPACE = {"units": [0, 128, 256, 512], "dropout": [0.0, 0.2, 0.4, 0.5], "l2": [0.0, 1e-4, 1e-3],
         "lr": (1e-4, 3e-3)}


def load_features(name: str):
    z = np.load(config.FEATURES_DIR / f"{name}.npz", allow_pickle=True)
    X, y, split = z["X"].astype(np.float32), z["y"].astype(np.int64), z["split"]
    return {s: (X[split == s], y[split == s]) for s in ("train", "val", "test")}, float(z["seconds"]) / len(y)


def keras_head(dim, units, dropout, l2, lr):
    import keras
    reg = keras.regularizers.l2(l2) if l2 else None
    layers = [keras.Input((dim,)), keras.layers.Dropout(dropout)]
    if units:
        layers += [keras.layers.Dense(units, activation="relu", kernel_regularizer=reg), keras.layers.Dropout(dropout)]
    layers += [keras.layers.Dense(1, activation="sigmoid", kernel_regularizer=reg)]
    m = keras.Sequential(layers)
    m.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy", metrics=["accuracy"])
    return m


def random_search(d, n_trials=16, seed=config.SEED):
    import keras
    keras.utils.set_random_seed(seed)
    rng = np.random.default_rng(seed)
    (Xtr, ytr), (Xva, yva) = d["train"], d["val"]
    trials, best = [], None
    for i in range(n_trials):
        hp = {"units": int(rng.choice(SPACE["units"])), "dropout": float(rng.choice(SPACE["dropout"])),
              "l2": float(rng.choice(SPACE["l2"])), "lr": float(np.exp(rng.uniform(*np.log(SPACE["lr"]))))}
        m = keras_head(Xtr.shape[1], **hp)
        stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)
        m.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=40, batch_size=64, callbacks=[stop], verbose=0)
        p = m.predict(Xva, verbose=0).ravel()
        rec = {**hp, "trial": i, "val_loss": float(log_loss(yva, np.clip(p, 1e-7, 1 - 1e-7))),
               "val_accuracy": float(((p >= 0.5) == yva).mean()), "epochs": len(m.history.history["loss"])}
        trials.append(rec)
        if best is None or rec["val_loss"] < best[0]["val_loss"]:
            best = (rec, m)
    return best, trials


def grid_logreg(d):
    (Xtr, ytr), (Xva, yva) = d["train"], d["val"]
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))
    gs = GridSearchCV(pipe, {"logisticregression__C": [0.001, 0.01, 0.1, 1.0]}, scoring="neg_log_loss",
                      cv=StratifiedKFold(5, shuffle=True, random_state=config.SEED), n_jobs=1)
    gs.fit(Xtr, ytr)
    p = gs.predict_proba(Xva)[:, 1]
    return gs, {"C": gs.best_params_["logisticregression__C"], "cv_log_loss": float(-gs.best_score_),
                "val_loss": float(log_loss(yva, p)), "val_accuracy": float(((p >= 0.5) == yva).mean())}


def run(name: str) -> dict:
    t0 = time.time()
    d, sec_per_img = load_features(name)
    gs, lr_res = grid_logreg(d)
    (best, model), trials = random_search(d)
    Xte, yte = d["test"]
    use_keras = best["val_loss"] <= lr_res["val_loss"]
    p_test = model.predict(Xte, verbose=0).ravel() if use_keras else gs.predict_proba(Xte)[:, 1]
    out = {"backbone": name, "input_size": config.BACKBONES[name], "feature_dim": int(Xte.shape[1]),
           "extract_ms_per_image": 1000 * sec_per_img, "logreg_grid": lr_res, "random_search_best": best,
           "random_search_trials": trials, "selected_head": "keras" if use_keras else "logreg",
           "search_seconds": time.time() - t0}
    np.save(config.REPORTS_DIR / "predictions" / f"{name}_test.npy", p_test)
    if use_keras:   # tête retenue conservée : sert à l'export du modèle complet (réseau gelé + tête)
        (config.ROOT / "data" / "cache").mkdir(parents=True, exist_ok=True)
        model.save(config.ROOT / "data" / "cache" / f"head_{name}.keras")
    return out


def main(names=None):
    (config.REPORTS_DIR / "predictions").mkdir(parents=True, exist_ok=True)
    f = config.REPORTS_DIR / "transfer_search.json"
    res = json.loads(f.read_text()) if names and f.exists() else {}
    for n in names or config.BACKBONES:
        res[n] = run(n)
    res = {n: res[n] for n in config.BACKBONES if n in res}
    f.write_text(json.dumps(res, indent=2))
    for n, r in res.items():
        b = r["random_search_best"]
        print(f"{n}: logreg val acc {r['logreg_grid']['val_accuracy']:.4f} | head val acc {b['val_accuracy']:.4f} "
              f"(units {b['units']}, dropout {b['dropout']}, lr {b['lr']:.1e}) -> {r['selected_head']}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1:] or None)
