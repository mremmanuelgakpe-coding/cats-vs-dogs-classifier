"""Échantillon : équilibre des classes et absence de fuite entre entraînement, validation et test."""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_manifest_splits():
    m = pd.read_csv(ROOT / "data" / "processed" / "manifest.csv")
    counts = m.groupby(["split", "label"]).size()
    assert counts[("train", "cat")] == counts[("train", "dog")] == 4000
    assert counts[("test", "cat")] == counts[("test", "dog")] == 1000
    # une même photo (copie exacte ou quasi identique) n'apparaît jamais dans deux échantillons
    assert not m["sha256"].duplicated().any() and not m["phash"].duplicated().any()
    assert set(m["label"]) == {"cat", "dog"} and ((m["label"] == "dog") == (m["y"] == 1)).all()


def test_quality_report():
    q = json.loads((ROOT / "reports" / "data_quality.json").read_text())
    assert q["files"] == 24989 and q["unique_images"] <= q["files"]
