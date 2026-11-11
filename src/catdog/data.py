"""Inventaire, contrôle qualité, déduplication et découpage des images.

Étapes :
1. inventaire des 24 989 photos du miroir (l'étiquette vient du nom : ``cat.123.jpg`` / ``dog.45.jpg``) ;
2. contrôle : fichier lisible par Pillow, dimensions minimales, conversion RVB possible ;
3. doublons : empreinte SHA-256 (copies exactes) et empreinte perceptuelle pHash (copies redimensionnées ou
   recompressées). Les images d'un même groupe restent dans le même échantillon (pas de fuite train → test),
   les groupes à étiquettes contradictoires sont écartés ;
4. échantillon stratifié de ``SAMPLE_PER_CLASS`` photos par classe, puis découpage train / val / test par groupe.

Sortie : ``data/processed/manifest.csv`` (chemin relatif, classe, empreintes, taille, échantillon). Aucune image
n'est copiée ni publiée.

Usage : PYTHONPATH=src python -m catdog.data
"""
from __future__ import annotations

import hashlib
import json
import warnings

import imagehash
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

from . import config

Image.MAX_IMAGE_PIXELS = 40_000_000  # refuse les « bombes de décompression »
MIN_SIDE = 32


def inventory() -> pd.DataFrame:
    rows = []
    for path in sorted(config.DATA_SRC.rglob("*.jpg")):
        name = path.name.lower()
        label = "cat" if name.startswith("cat.") else "dog" if name.startswith("dog.") else None
        folder = path.parent.name.rstrip("s")
        rows.append({"path": str(path.relative_to(config.DATA_SRC)), "label": label, "folder_label": folder})
    return pd.DataFrame(rows)


def inspect(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for rel in df["path"]:
        p = config.DATA_SRC / rel
        rec = {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "width": 0, "height": 0, "phash": "", "issue": ""}
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(p) as im:
                    im.verify()
                with Image.open(p) as im:
                    im = im.convert("RGB")
                    rec["width"], rec["height"] = im.size
                    rec["phash"] = str(imagehash.phash(im))
            if min(rec["width"], rec["height"]) < MIN_SIDE:
                rec["issue"] = "trop petite"
        except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombWarning) as exc:
            rec["issue"] = f"illisible ({type(exc).__name__})"
        out.append(rec)
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(out)], axis=1)


def build() -> dict:
    df = inspect(inventory())
    report = {"files": int(len(df)), "label_mismatch_folder": int((df["label"] != df["folder_label"]).sum()),
              "unreadable_or_small": int((df["issue"] != "").sum())}
    df = df[(df["issue"] == "") & df["label"].notna()].copy()

    # Groupes de doublons : même pHash (couvre aussi les copies exactes)
    df["group"] = df.groupby("phash").ngroup()
    labels_per_group = df.groupby("group")["label"].nunique()
    conflict = labels_per_group[labels_per_group > 1].index
    report["exact_duplicates"] = int(df["sha256"].duplicated().sum())
    report["near_duplicate_groups"] = int((df.groupby("group").size() > 1).sum())
    report["conflicting_label_images"] = int(df["group"].isin(conflict).sum())
    df = df[~df["group"].isin(conflict)]
    # Une seule image par groupe : les doublons n'apportent rien et biaiseraient l'évaluation
    df = df.sort_values("path").drop_duplicates("group", keep="first")
    report["unique_images"] = int(len(df))
    report["unique_by_class"] = df["label"].value_counts().to_dict()

    rng = np.random.default_rng(config.SEED)
    parts = []
    for label in config.CLASSES:
        sub = df[df["label"] == label]
        idx = rng.permutation(len(sub))[: config.SAMPLE_PER_CLASS]
        sub = sub.iloc[idx].copy()
        n = len(sub)
        n_tr, n_va = round(n * config.SPLIT["train"]), round(n * config.SPLIT["val"])
        sub["split"] = ["train"] * n_tr + ["val"] * n_va + ["test"] * (n - n_tr - n_va)
        parts.append(sub)
    m = pd.concat(parts).sort_values(["split", "label", "path"]).reset_index(drop=True)
    m["y"] = (m["label"] == "dog").astype(int)
    cols = ["path", "label", "y", "split", "width", "height", "sha256", "phash"]
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    m[cols].to_csv(config.PROCESSED_DIR / "manifest.csv", index=False)
    report["sample"] = m.groupby(["split", "label"]).size().unstack().to_dict(orient="index")
    report["median_size"] = [int(m["width"].median()), int(m["height"].median())]
    report["data_commit"] = config.DATA_COMMIT
    (config.REPORTS_DIR / "data_quality.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def load_manifest(split: str | None = None) -> pd.DataFrame:
    m = pd.read_csv(config.PROCESSED_DIR / "manifest.csv")
    return m if split is None else m[m["split"] == split].reset_index(drop=True)


if __name__ == "__main__":
    print(json.dumps(build(), indent=2, ensure_ascii=False))
