"""
Integration tests for the FastAPI pronunciation assessment endpoints.
Uses httpx TestClient; does NOT load real model weights (scorer is mocked).
"""

import base64
import io
import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from src.models.base_scorer import PhonemeResult, ScoringResult, WordResult


def _make_wav_bytes(duration_sec: float = 0.5, sr: int = 16000) -> bytes:
    """Generate a silent WAV file in memory."""
    samples = np.zeros(int(duration_sec * sr), dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, samples, sr, format="WAV")
    buf.seek(0)
    return buf.read()


MOCK_RESULT = ScoringResult(
    recognition_status="Success",
    display_text="Hello",
    offset_ms=0,
    duration_ms=500,
    pron_score=85.0,
    accuracy_score=88.0,
    fluency_score=78.0,
    completeness_score=100.0,
    words=[
        WordResult(
            word="hello",
            accuracy_score=88.0,
            error_type="None",
            offset_ms=0,
            duration_ms=500,
            phonemes=[
                PhonemeResult("h", 90.0, 0, 80),
                PhonemeResult("ɛ", 85.0, 80, 80),
                PhonemeResult("l", 88.0, 160, 80),
                PhonemeResult("o", 87.0, 240, 80),
            ],
        )
    ],
)


@pytest.fixture
def client():
    """
    Create a test client with a mocked scorer.

    We pre-set server._scorer before the TestClient context manager opens
    (which triggers lifespan startup). The lifespan checks if _scorer is
    already set and skips model loading, so torch is never imported.
    """
    from src.api import server
    from src.api.server import app

    mock_scorer = MagicMock()
    mock_scorer.model.config.model_type = "wav2vec2"
    mock_scorer.device = "cpu"
    mock_scorer.score.return_value = MOCK_RESULT

    # Inject BEFORE the context manager starts (before lifespan fires)
    server._scorer = mock_scorer
    try:
        with TestClient(app) as c:
            yield c
    finally:
        server._scorer = None


class TestHealthEndpoint:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"


class TestFileEndpoint:
    def test_score_valid_wav(self, client):
        wav = _make_wav_bytes()
        resp = client.post(
            "/pronunciation-assessment/file",
            data={"reference_text": "hello"},
            files={"audio_file": ("hello.wav", wav, "audio/wav")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "NBest" in body
        assert body["NBest"][0]["PronScore"] == pytest.approx(85.0)

    def test_rejects_oversized_audio(self, client):
        wav = _make_wav_bytes(duration_sec=31.0)
        resp = client.post(
            "/pronunciation-assessment/file",
            data={"reference_text": "hello"},
            files={"audio_file": ("long.wav", wav, "audio/wav")},
        )
        assert resp.status_code == 413


class TestJsonEndpoint:
    def test_score_base64_audio(self, client):
        wav = _make_wav_bytes()
        encoded = base64.b64encode(wav).decode()
        payload = {
            "reference_text": "hello",
            "audio_base64": encoded,
            "grading_system": "HundredMark",
            "granularity": "Phoneme",
            "phoneme_alphabet": "IPA",
            "enable_miscue": False,
        }
        resp = client.post("/pronunciation-assessment/json", json=payload)
        assert resp.status_code == 200
        body = resp.json()
        assert body["RecognitionStatus"] == "Success"
        assert len(body["NBest"][0]["Words"]) == 1

    def test_invalid_base64_returns_400(self, client):
        payload = {
            "reference_text": "hello",
            "audio_base64": "!!!not_valid_base64!!!",
            "grading_system": "HundredMark",
            "granularity": "Phoneme",
            "phoneme_alphabet": "IPA",
            "enable_miscue": False,
        }
        resp = client.post("/pronunciation-assessment/json", json=payload)
        assert resp.status_code == 400
