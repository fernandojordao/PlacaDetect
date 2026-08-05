"""Smoke test do pipeline completo: upload -> processamento -> resultado.

Usa uma imagem sintética (sem placa real) apenas para validar a integração
entre upload, detecção, redação, geração de thumbnails e persistência no
banco. A qualidade da detecção em si já é validada pelo modelo pré-treinado
(open-image-models / YOLOv9), não é o foco deste teste.
"""

import io
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import pytest
from PIL import Image

TEST_DATA_DIR = Path(tempfile.mkdtemp(prefix="placadetect_test_"))
os.environ["PLACADETECT_DATA_DIR"] = str(TEST_DATA_DIR)

BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _fake_jpeg_bytes(color=(120, 140, 160)) -> bytes:
    img = Image.new("RGB", (640, 480), color=color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_upload_process_and_list(client):
    files = [("files", ("teste1.jpg", _fake_jpeg_bytes(), "image/jpeg"))]
    res = client.post("/api/upload", files=files)
    assert res.status_code == 200
    body = res.json()
    assert body["added"] == 1
    photo_id = body["photo_ids"][0]

    res = client.post("/api/process", json={"scope": "pending"})
    assert res.status_code == 200
    assert res.json()["started"] is True

    for _ in range(60):
        status = client.get("/api/process/status").json()
        if not status["active"]:
            break
        time.sleep(0.5)
    else:
        raise AssertionError("processamento não concluiu a tempo")

    photo = client.get(f"/api/photos/{photo_id}").json()
    assert photo["status"] in ("success", "no_plate")
    assert photo["output_url"] is not None
    assert Path(photo["output_path"]).exists()
    assert photo["thumb_output_url"] is not None

    stats = client.get("/api/stats").json()
    assert stats["total"] == 1


def teardown_module(module):
    shutil.rmtree(TEST_DATA_DIR, ignore_errors=True)
