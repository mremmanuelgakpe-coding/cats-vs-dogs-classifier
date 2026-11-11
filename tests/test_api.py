"""API : authentification, contrôles d'entrée, réponse et en-têtes de sécurité."""
import importlib

import pytest
from fastapi.testclient import TestClient

from conftest import make_image, needs_model

pytestmark = needs_model
KEY = "cle-de-test-suffisamment-longue"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("API_KEY", KEY)
    monkeypatch.delenv("ENABLE_DOCS", raising=False)
    import api.main as main
    importlib.reload(main)
    with TestClient(main.app) as c:
        yield c


def post(client, data, key=KEY, name="photo.jpg", explain=False):
    return client.post(f"/predict?explain={str(explain).lower()}", files={"image": (name, data, "image/jpeg")},
                       headers={"X-API-Key": key} if key else {})


def test_health_is_public(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_api_key_required(client, jpeg):
    assert post(client, jpeg, key=None).status_code == 401
    assert post(client, jpeg, key="mauvaise").status_code == 401
    assert client.get("/model-info").status_code == 401


def test_missing_server_key(monkeypatch, jpeg):
    monkeypatch.delenv("API_KEY", raising=False)
    import api.main as main
    importlib.reload(main)
    with TestClient(main.app) as c:
        assert post(c, jpeg).status_code == 503


def test_prediction_and_explanation(client, jpeg):
    r = post(client, jpeg, explain=True)
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in {"cat", "dog", "unsure"} and len(body["occlusion"]) == 8
    assert r.headers["x-content-type-options"] == "nosniff" and r.headers["cache-control"] == "no-store"


def test_bad_inputs(client):
    assert post(client, make_image("GIF"), name="x.gif").status_code == 415
    assert post(client, b"du texte deguise en jpeg").status_code == 422
    assert post(client, b"\x00" * (9 * 1024 * 1024)).status_code == 413


def test_docs_disabled_by_default(client):
    assert client.get("/docs").status_code == 404
    assert client.get("/openapi.json").status_code == 404
