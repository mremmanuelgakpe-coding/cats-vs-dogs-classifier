"""Démo Streamlit : chaque page s'exécute sans erreur, sans Internet et sans photo."""
from pathlib import Path

import pytest

from conftest import needs_model

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

pytestmark = needs_model
APP = str(Path(__file__).resolve().parents[1] / "demo" / "app.py")


def run(at):
    at.run(timeout=120)
    assert not at.exception, at.exception
    return at


@pytest.mark.parametrize("page", ["views/classify.py", "views/performance.py", "views/method.py"])
def test_pages(page):
    at = run(AppTest.from_file(APP, default_timeout=120))
    at.switch_page(page)
    run(at)
    assert any("kgrid" in m.value for m in at.markdown)
