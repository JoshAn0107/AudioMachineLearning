#!/usr/bin/env python3
"""
Benchmark SpeakRight with diverse audio quality levels.

Tests 5 quality tiers + wrong-word variants against both SpeakRight and Azure.
Produces:
  - results/benchmark_diverse.csv
  - results/diverse_comparison_plot.png
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import soundfile as sf
from pydub import AudioSegment
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

AZURE_KEY = os.getenv("AZURE_SPEECH_KEY", "")
AZURE_REGION = os.getenv("AZURE_SPEECH_REGION", "")
DIVERSE_DIR = Path("data/diverse_audio")
RESULTS_DIR = Path("results")


def score_with_speakright(audio_path: Path, reference_text: str, scorer) -> dict | None:
    try:
        audio, sr = sf.read(str(audio_path), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        result = scorer.score(audio, reference_text, sample_rate=sr)
        return {
            "pron_score": result.pron_score,
            "accuracy_score": result.accuracy_score,
            "fluency_score": result.fluency_score,
            "completeness_score": result.completeness_score,
            "recognised_text": result.display_text,
        }
    except Exception as e:
        logger.error("SpeakRight error for '%s': %s", reference_text, e)
        return None


def score_with_azure(audio_path: Path, reference_text: str) -> dict | None:
    if not AZURE_KEY or not AZURE_REGION:
        return None
    try:
        import azure.cognitiveservices.speech as speechsdk
        speech_config = speechsdk.SpeechConfig(subscription=AZURE_KEY, region=AZURE_REGION)
        speech_config.speech_recognition_language = "en-US"
        pron_config = speechsdk.PronunciationAssessmentConfig(
            reference_text=reference_text,
            grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
            granularity=speechsdk.PronunciationAssessmentGranularity.Phoneme,
            enable_miscue=True,
        )
        audio_config = speechsdk.audio.AudioConfig(filename=str(audio_path))
        recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
        pron_config.apply_to(recognizer)
        result = recognizer.recognize_once()
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            pa = speechsdk.PronunciationAssessmentResult(result)
            return {
                "pron_score": pa.pronunciation_score,
                "accuracy_score": pa.accuracy_score,
                "fluency_score": pa.fluency_score,
                "completeness_score": pa.completeness_score,
                "recognised_text": result.text,
            }
        return None
    except Exception as e:
        logger.error("Azure error: %s", e)
        return None


# Test cases: (reference_text, audio_filename, quality_tier)
TEST_WORDS = ["hello", "beautiful", "technology", "apple", "environment",
              "congratulations", "university", "responsibility"]
TEST_SENTENCES = [
    "The quick brown fox jumps over the lazy dog",
    "She sells seashells by the seashore",
    "I would like a cup of coffee please",
]
QUALITIES = ["native", "good", "medium", "poor", "wrong"]
MISREAD_PAIRS = [
    ("hello", "hello_misread"),
    ("apple", "apple_misread"),
    ("university", "university_misread"),
    ("beautiful", "beautiful_misread"),
]


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load scorer
    sys.path.insert(0, ".")
    from src.models.wav2vec2_scorer import Wav2Vec2PronunciationScorer
    model_name = os.getenv("SPEAKRIGHT_MODEL", "facebook/wav2vec2-large-960h")
    scorer = Wav2Vec2PronunciationScorer(model_name=model_name)
    logger.info("Model loaded: %s on %s", model_name, scorer.device)

    records = []
    all_texts = TEST_WORDS + TEST_SENTENCES

    # Score each quality variant
    for text in all_texts:
        safe_name = text.lower().replace(" ", "_")[:40]
        text_type = "word" if text in TEST_WORDS else "sentence"

        for quality in QUALITIES:
            audio_path = DIVERSE_DIR / f"{safe_name}_{quality}.wav"
            if not audio_path.exists():
                logger.warning("Missing: %s", audio_path)
                continue

            logger.info("[%s/%s] %s", quality, text_type, text[:30])

            sr = score_with_speakright(audio_path, text, scorer)
            az = score_with_azure(audio_path, text)

            record = {
                "text": text,
                "type": text_type,
                "quality": quality,
                "speakright_pron": sr["pron_score"] if sr else None,
                "speakright_acc": sr["accuracy_score"] if sr else None,
                "speakright_flu": sr["fluency_score"] if sr else None,
                "speakright_comp": sr["completeness_score"] if sr else None,
                "speakright_rec": sr["recognised_text"] if sr else None,
                "azure_pron": az["pron_score"] if az else None,
                "azure_acc": az["accuracy_score"] if az else None,
                "azure_flu": az["fluency_score"] if az else None,
                "azure_comp": az["completeness_score"] if az else None,
                "azure_rec": az["recognised_text"] if az else None,
            }
            records.append(record)

            if sr and az:
                logger.info("  SR=%5.1f  AZ=%5.1f  (acc SR=%5.1f AZ=%5.1f)",
                            sr["pron_score"], az["pron_score"],
                            sr["accuracy_score"], az["accuracy_score"])

    # Score misread variants (say wrong word for reference)
    for ref_word, audio_name in MISREAD_PAIRS:
        audio_path = DIVERSE_DIR / f"{audio_name}.wav"
        if not audio_path.exists():
            continue
        logger.info("[misread] ref='%s' audio='%s'", ref_word, audio_name)
        sr = score_with_speakright(audio_path, ref_word, scorer)
        az = score_with_azure(audio_path, ref_word)
        records.append({
            "text": ref_word,
            "type": "word",
            "quality": "misread",
            "speakright_pron": sr["pron_score"] if sr else None,
            "speakright_acc": sr["accuracy_score"] if sr else None,
            "speakright_flu": sr["fluency_score"] if sr else None,
            "speakright_comp": sr["completeness_score"] if sr else None,
            "speakright_rec": sr["recognised_text"] if sr else None,
            "azure_pron": az["pron_score"] if az else None,
            "azure_acc": az["accuracy_score"] if az else None,
            "azure_flu": az["fluency_score"] if az else None,
            "azure_comp": az["completeness_score"] if az else None,
            "azure_rec": az["recognised_text"] if az else None,
        })

    # Save CSV
    df = pd.DataFrame(records)
    csv_path = RESULTS_DIR / "benchmark_diverse.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved %d rows to %s", len(df), csv_path)

    # Print summary by quality tier
    print("\n" + "=" * 80)
    print("DIVERSE BENCHMARK SUMMARY")
    print("=" * 80)
    for q in QUALITIES + ["misread"]:
        qdf = df[df["quality"] == q].dropna(subset=["speakright_pron"])
        if qdf.empty:
            continue
        sr_pron = qdf["speakright_pron"].mean()
        sr_acc = qdf["speakright_acc"].mean()
        az_pron = qdf["azure_pron"].mean() if qdf["azure_pron"].notna().any() else float("nan")
        az_acc = qdf["azure_acc"].mean() if qdf["azure_acc"].notna().any() else float("nan")
        print(f"  {q:10s}  n={len(qdf):2d}  SR_pron={sr_pron:5.1f}  SR_acc={sr_acc:5.1f}  AZ_pron={az_pron:5.1f}  AZ_acc={az_acc:5.1f}")
    print("=" * 80)

    # Generate plot
    plot_diverse(df)
    logger.info("Benchmark complete!")


def plot_diverse(df: pd.DataFrame):
    """Generate a grouped bar chart by quality tier."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    quality_order = ["native", "good", "medium", "poor", "wrong", "misread"]
    quality_colors = {
        "native": "#2ecc71", "good": "#3498db", "medium": "#f39c12",
        "poor": "#e74c3c", "wrong": "#9b59b6", "misread": "#e67e22"
    }

    for ax, (metric, label) in zip(axes, [("pron", "Pronunciation Score"), ("acc", "Accuracy Score")]):
        sr_means = []
        az_means = []
        labels = []

        for q in quality_order:
            qdf = df[df["quality"] == q]
            if qdf.empty:
                continue
            sr_col = f"speakright_{metric}"
            az_col = f"azure_{metric}"
            sr_val = qdf[sr_col].dropna().mean() if qdf[sr_col].notna().any() else 0
            az_val = qdf[az_col].dropna().mean() if qdf[az_col].notna().any() else 0
            sr_means.append(sr_val)
            az_means.append(az_val)
            labels.append(q.capitalize())

        x = np.arange(len(labels))
        width = 0.35

        bars1 = ax.bar(x - width/2, sr_means, width, label="SpeakRight", color="#4A90D9", edgecolor="k", linewidth=0.5)
        bars2 = ax.bar(x + width/2, az_means, width, label="Azure", color="#E8A838", edgecolor="k", linewidth=0.5)

        ax.set_ylabel("Score (0-100)", fontsize=11)
        ax.set_title(f"{label} by Quality Tier", fontsize=13, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylim(0, 110)
        ax.legend(fontsize=10)
        ax.grid(axis="y", alpha=0.3)

        # Add value labels
        for bar in bars1:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2., h + 1, f"{h:.0f}", ha="center", va="bottom", fontsize=8)
        for bar in bars2:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2., h + 1, f"{h:.0f}", ha="center", va="bottom", fontsize=8)

    plt.suptitle("SpeakRight vs Azure: Diverse Audio Quality Benchmark",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(str(RESULTS_DIR / "diverse_comparison_plot.png"), dpi=150, bbox_inches="tight")
    logger.info("Plot saved")


if __name__ == "__main__":
    main()
