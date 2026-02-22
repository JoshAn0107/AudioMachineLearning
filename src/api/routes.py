"""
FastAPI route definitions for the SpeakRight pronunciation assessment API.

Endpoints:
  GET  /health                         – liveness check
  POST /pronunciation-assessment/file  – score from uploaded WAV file
  POST /pronunciation-assessment/json  – score from base64-encoded audio + JSON params
"""

import base64
import io
import logging

import numpy as np
import soundfile as sf
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from .schemas import (
    HealthResponse,
    PronunciationAssessmentRequest,
    PronunciationAssessmentResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


def get_scorer():
    """Dependency-injected scorer; populated at startup by server.py."""
    from .server import _scorer
    if _scorer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    return _scorer


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse)
def health(scorer=Depends(get_scorer)):
    return HealthResponse(
        status="ok",
        model=scorer.model.config.model_type,
        device=str(scorer.device),
    )


# ---------------------------------------------------------------------------
# Score from uploaded file (multipart/form-data)
# Mirrors: POST /speech/recognition/conversation with a WAV attachment
# ---------------------------------------------------------------------------

@router.post("/pronunciation-assessment/file", response_model=PronunciationAssessmentResponse)
async def assess_from_file(
    audio_file: UploadFile = File(..., description="WAV or FLAC file, mono, 16 kHz"),
    reference_text: str = Form(...),
    granularity: str = Form("Phoneme"),
    scorer=Depends(get_scorer),
):
    raw = await audio_file.read()
    audio, sr = _read_audio_bytes(raw)
    return _run_scoring(scorer, audio, sr, reference_text)


# ---------------------------------------------------------------------------
# Score from JSON body with base64 audio
# ---------------------------------------------------------------------------

class AudioJsonRequest(PronunciationAssessmentRequest):
    audio_base64: str  # base64-encoded WAV bytes
    sample_rate: int = 16000


@router.post("/pronunciation-assessment/json", response_model=PronunciationAssessmentResponse)
def assess_from_json(request: AudioJsonRequest, scorer=Depends(get_scorer)):
    try:
        raw = base64.b64decode(request.audio_base64)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid base64 audio: {exc}")
    audio, sr = _read_audio_bytes(raw)
    return _run_scoring(scorer, audio, sr, request.reference_text)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _read_audio_bytes(raw: bytes) -> tuple[np.ndarray, int]:
    try:
        buf = io.BytesIO(raw)
        audio, sr = sf.read(buf, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        return audio, sr
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Cannot decode audio: {exc}")


def _run_scoring(scorer, audio: np.ndarray, sr: int, reference_text: str) -> dict:
    if len(audio) / sr > 30:
        raise HTTPException(status_code=413, detail="Audio exceeds 30-second limit.")
    try:
        result = scorer.score(audio, reference_text, sample_rate=sr)
        return result.to_azure_format()
    except Exception as exc:
        logger.exception("Scoring failed")
        raise HTTPException(status_code=500, detail=str(exc))
