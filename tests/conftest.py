import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
MODELS = ROOT / "models"


def make_image(fmt="JPEG", size=(320, 240), seed=0) -> bytes:
    """Image synthétique (aucune photo du jeu de données dans les tests)."""
    rng = np.random.default_rng(seed)
    arr = (rng.random((size[1], size[0], 3)) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture
def jpeg():
    return make_image()


needs_model = pytest.mark.skipif(not (MODELS / "model.onnx").exists(), reason="modèle ONNX non généré")
