"""
Side-by-side comparison of SpeakRight vs Azure Pronunciation Assessment.

Requires:
    - AZURE_SPEECH_KEY and AZURE_SPEECH_REGION in environment (or .env file)
    - azure-cognitiveservices-speech installed

Run:
    python -m src.evaluation.azure_compare --words data/test_words.txt --audio-dir data/recordings/
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from ..features.extractor import load_audio
from ..models.wav2vec2_scorer import Wav2Vec2PronunciationScorer
from .metrics import summarise_comparison, error_detection_metrics

load_dotenv()
logger = logging.getLogger(__name__)


def score_with_azure(audio_path: Path, reference_text: str) -> dict | None:
    """
    Call Azure Pronunciation Assessment SDK for a single audio file.
    Returns the full Azure response dict or None on failure.
    """
    try:
        import azure.cognitiveservices.speech as speechsdk
    except ImportError:
        logger.error("azure-cognitiveservices-speech not installed. Run: pip install azure-cognitiveservices-speech")
        return None

    key = os.getenv("AZURE_SPEECH_KEY")
    region = os.getenv("AZURE_SPEECH_REGION")
    if not key or not region:
        raise EnvironmentError("Set AZURE_SPEECH_KEY and AZURE_SPEECH_REGION in your .env file.")

    speech_config = speechsdk.SpeechConfig(subscription=key, region=region)
    speech_config.speech_recognition_language = "en-US"

    pron_config = speechsdk.PronunciationAssessmentConfig(
        reference_text=reference_text,
        grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
        granularity=speechsdk.PronunciationAssessmentGranularity.Phoneme,
        enable_miscue=True,
    )
    pron_config.phoneme_alphabet = "IPA"

    audio_config = speechsdk.AudioConfig(filename=str(audio_path))
    recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    pron_config.apply_to(recognizer)

    result = recognizer.recognize_once_async().get()
    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        pa_result = speechsdk.PronunciationAssessmentResult(result)
        return {
            "pron_score": pa_result.pronunciation_score,
            "accuracy_score": pa_result.accuracy_score,
            "fluency_score": pa_result.fluency_score,
            "completeness_score": pa_result.completeness_score,
            "words": [
                {
                    "word": w.word,
                    "accuracy_score": w.accuracy_score,
                    "error_type": str(w.error_type).split(".")[-1],
                }
                for w in pa_result.words
            ],
        }
    logger.warning("Azure recognition failed for %s: %s", audio_path.name, result.reason)
    return None


def run_benchmark(
    audio_dir: str | Path,
    word_list: list[str],
    output_csv: str | Path = "results/benchmark.csv",
    speakright_model: str = "facebook/wav2vec2-base",
) -> pd.DataFrame:
    """
    For each word in word_list, find a matching WAV in audio_dir, score with
    both SpeakRight and Azure, and return a comparison DataFrame.

    Audio files must be named exactly as the word (e.g., "hello.wav").
    """
    audio_dir = Path(audio_dir)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    scorer = Wav2Vec2PronunciationScorer(model_name=speakright_model)
    records = []

    for word in word_list:
        audio_path = audio_dir / f"{word.lower()}.wav"
        if not audio_path.exists():
            logger.warning("Audio not found: %s — skipping.", audio_path)
            continue

        audio, sr = load_audio(audio_path)

        # SpeakRight
        sr_result = scorer.score(audio, word, sample_rate=sr)

        # Azure
        az_result = score_with_azure(audio_path, word)
        if az_result is None:
            continue

        records.append(
            {
                "word": word,
                "speakright_pron": sr_result.pron_score,
                "azure_pron": az_result["pron_score"],
                "speakright_acc": sr_result.accuracy_score,
                "azure_acc": az_result["accuracy_score"],
                "speakright_fluency": sr_result.fluency_score,
                "azure_fluency": az_result["fluency_score"],
                "speakright_completeness": sr_result.completeness_score,
                "azure_completeness": az_result["completeness_score"],
            }
        )
        logger.info(
            "%-20s  SR=%.1f  AZ=%.1f",
            word,
            sr_result.pron_score,
            az_result["pron_score"],
        )

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    logger.info("Saved %d rows to %s", len(df), output_csv)

    if not df.empty:
        summary = summarise_comparison(records)
        logger.info("Summary: %s", json.dumps(summary, indent=2))

    return df


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark SpeakRight vs Azure.")
    parser.add_argument("--words", required=True, help="Text file with one word per line.")
    parser.add_argument("--audio-dir", required=True, help="Directory containing <word>.wav files.")
    parser.add_argument("--output", default="results/benchmark.csv")
    parser.add_argument("--model", default="facebook/wav2vec2-base")
    args = parser.parse_args()

    word_list = Path(args.words).read_text().strip().splitlines()
    run_benchmark(
        audio_dir=args.audio_dir,
        word_list=word_list,
        output_csv=args.output,
        speakright_model=args.model,
    )
