"""Décodage sûr des images envoyées et prédicteur ONNX."""
import io
import json
import shutil

import numpy as np
import pytest
from PIL import Image

from catdog.inference import MAX_BYTES, ImageError, Predictor, decode
from conftest import MODELS, make_image, needs_model


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP"])
def test_accepted_formats(fmt):
    im = decode(make_image(fmt))
    assert im.mode == "RGB" and im.size == (320, 240)


@pytest.mark.parametrize("data, message", [
    (b"", "vide"),
    (b"pas une image du tout", "illisible"),
    (b"\x00" * (MAX_BYTES + 1), "lourde"),
])
def test_rejected_payloads(data, message):
    with pytest.raises(ImageError, match=message):
        decode(data)


def test_rejects_gif_and_tiny_images():
    with pytest.raises(ImageError, match="Format"):
        decode(make_image("GIF"))
    with pytest.raises(ImageError, match="petite"):
        decode(make_image("PNG", size=(20, 20)))


def test_rejects_decompression_bomb():
    buf = io.BytesIO()
    Image.new("L", (9000, 9000)).save(buf, format="PNG")      # 81 Mpx, quelques dizaines de Ko compressés
    assert len(buf.getvalue()) < MAX_BYTES
    with pytest.raises(ImageError):
        decode(buf.getvalue())


def test_exif_orientation_applied():
    im = Image.fromarray(np.zeros((100, 200, 3), np.uint8))
    exif = Image.Exif()
    exif[0x0112] = 6                                          # rotation 90°
    buf = io.BytesIO()
    im.save(buf, format="JPEG", exif=exif)
    assert decode(buf.getvalue()).size == (100, 200)


@needs_model
def test_predictor_outputs():
    p = Predictor(MODELS)
    r = p.predict(decode(make_image()))
    assert r.label in {"cat", "dog", "unsure"} and 0 <= r.probability_dog <= 1 and 0.5 <= r.confidence <= 1
    occ = p.occlusion(decode(make_image(seed=1)))
    assert occ.shape == (8, 8) and (occ >= 0).all()


@needs_model
def test_tampered_model_is_refused(tmp_path):
    for f in MODELS.iterdir():
        shutil.copy(f, tmp_path / f.name)
    meta = json.loads((tmp_path / "model.meta.json").read_text())
    meta["model_sha256"] = "0" * 64
    (tmp_path / "model.meta.json").write_text(json.dumps(meta))
    with pytest.raises(RuntimeError, match="Empreinte"):
        Predictor(tmp_path)
