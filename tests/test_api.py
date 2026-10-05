import os

import pytest
from fastapi.testclient import TestClient

from app.main import app
from tests.conftest import SHORT_SAMPLE

RESULT_KEYS = {"text", "segments", "audio_duration_s", "processing_s", "backend"}


@pytest.fixture(scope="module")
def client():
    os.environ["ASR_BACKEND"] = "fake"
    with TestClient(app) as c:
        yield c


def test_health_before_startup_is_503():
    assert TestClient(app).get("/health").status_code == 503


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "backend": "fake"}


def test_transcribe_sample_json_shape(client, tone_wav):
    data = SHORT_SAMPLE.read_bytes() if SHORT_SAMPLE.exists() else tone_wav
    r = client.post("/transcribe", files={"file": ("a.ogg", data, "audio/ogg")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == RESULT_KEYS
    assert body["backend"] == "fake"
    assert isinstance(body["segments"], list)
    for seg in body["segments"]:
        assert set(seg) == {"start", "end", "text"}


def test_transcribe_vad_false_covers_whole_file(client, tone_wav):
    r = client.post("/transcribe?vad=false", files={"file": ("a.wav", tone_wav, "audio/wav")})
    assert r.status_code == 200
    body = r.json()
    assert body["segments"][0]["start"] == 0.0
    assert body["segments"][-1]["end"] == body["audio_duration_s"]


def test_missing_file_is_400(client):
    assert client.post("/transcribe").status_code == 400
    assert client.post("/transcribe", data={"other": "x"}).status_code == 400


def test_empty_file_is_400(client):
    r = client.post("/transcribe", files={"file": ("a.wav", b"", "audio/wav")})
    assert r.status_code == 400


def test_random_bytes_is_400(client):
    r = client.post("/transcribe", files={"file": ("a.wav", os.urandom(2048), "audio/wav")})
    assert r.status_code == 400


def test_oversized_upload_is_413(client):
    limit = client.app.state.settings.max_upload_bytes
    r = client.post("/transcribe", files={"file": ("big.wav", b"\0" * (limit + 1), "audio/wav")})
    assert r.status_code == 413
