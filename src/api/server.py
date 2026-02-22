"""
SpeakRight FastAPI application entry point.

Run:
    uvicorn src.api.server:app --host 0.0.0.0 --port 8000 --reload

Environment variables:
    SPEAKRIGHT_MODEL   - HuggingFace model ID (default: facebook/wav2vec2-base)
    SPEAKRIGHT_DEVICE  - "cpu" | "cuda" (default: auto-detect)
"""

import logging
import os

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routes import router

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Module-level scorer instance shared across requests
_scorer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load the model once at startup; release on shutdown.

    If _scorer is already set (e.g. injected by tests), skip loading entirely
    so tests can run without torch/transformers installed.
    """
    global _scorer
    loaded_here = False

    if _scorer is None:
        model_name = os.getenv("SPEAKRIGHT_MODEL", "facebook/wav2vec2-base")
        device = os.getenv("SPEAKRIGHT_DEVICE", None)
        logger.info("Loading SpeakRight scorer: %s", model_name)
        from ..models.wav2vec2_scorer import Wav2Vec2PronunciationScorer
        _scorer = Wav2Vec2PronunciationScorer(model_name=model_name, device=device)
        logger.info("Model ready on %s", _scorer.device)
        loaded_here = True

    yield  # server runs here

    if loaded_here:
        logger.info("Shutting down SpeakRight scorer.")
        _scorer = None


app = FastAPI(
    title="SpeakRight Pronunciation Assessment API",
    description=(
        "Offline pronunciation scoring that mirrors Azure's Pronunciation Assessment API. "
        "Powered by wav2vec2 + GOP scoring."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8000, reload=True)
