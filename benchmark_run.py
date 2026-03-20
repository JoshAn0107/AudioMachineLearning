#!/usr/bin/env python3
"""
SpeakRight vs Azure Pronunciation Assessment Benchmark.

Generates speech audio via Google TTS (native-quality reference),
then scores each sample with both SpeakRight (offline wav2vec2+GOP)
and Azure Speech Service for side-by-side comparison.

Outputs:
  - results/benchmark.csv      (per-word scores from both systems)
  - results/comparison_plot.png (scatter plot: Azure vs SpeakRight)
"""

import json
import logging
import os
import sys
import time
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import soundfile as sf
from pydub import AudioSegment

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

AZURE_KEY = os.getenv("AZURE_SPEECH_KEY", "")
AZURE_REGION = os.getenv("AZURE_SPEECH_REGION", "")

# Test words: mix of easy, medium, hard pronunciation
WORD_LIST = [
    "hello",
    "world",
    "beautiful",
    "pronunciation",
    "technology",
    "comfortable",
    "environment",
    "vocabulary",
    "communication",
    "international",
    "apple",
    "banana",
    "elephant",
    "university",
    "development",
    "extraordinary",
    "congratulations",
    "sophisticated",
    "determination",
    "responsibility",
]

# Also test some sentences
SENTENCE_LIST = [
    "The quick brown fox jumps over the lazy dog",
    "She sells seashells by the seashore",
    "How much wood would a woodchuck chuck",
    "Peter Piper picked a peck of pickled peppers",
    "I would like a cup of coffee please",
]

AUDIO_DIR = Path("data/benchmark_audio")
RESULTS_DIR = Path("results")


# ---------------------------------------------------------------------------
# Audio generation via gTTS
# ---------------------------------------------------------------------------

def generate_tts_audio(text: str, output_path: Path) -> bool:
    """Generate speech audio using Google TTS and convert to 16kHz WAV."""
    try:
        from gtts import gTTS

        tts = gTTS(text=text, lang="en", slow=False)

        # gTTS outputs MP3; convert to 16kHz mono WAV
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_mp3 = tmp.name
            tts.save(tmp_mp3)

        audio = AudioSegment.from_mp3(tmp_mp3)
        audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
        audio.export(str(output_path), format="wav", parameters=["-acodec", "pcm_s16le"])

        os.remove(tmp_mp3)
        return True
    except Exception as e:
        logger.error("TTS generation failed for '%s': %s", text, e)
        return False


# ---------------------------------------------------------------------------
# SpeakRight scoring (local wav2vec2 + GOP)
# ---------------------------------------------------------------------------

def score_with_speakright(audio_path: Path, reference_text: str, scorer) -> dict | None:
    """Score audio with the local SpeakRight wav2vec2 scorer."""
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
        logger.error("SpeakRight scoring failed for '%s': %s", reference_text, e)
        return None


# ---------------------------------------------------------------------------
# Azure scoring
# ---------------------------------------------------------------------------

def score_with_azure(audio_path: Path, reference_text: str) -> dict | None:
    """Score audio with Azure Pronunciation Assessment SDK."""
    try:
        import azure.cognitiveservices.speech as speechsdk
    except ImportError:
        logger.error("Azure SDK not available")
        return None

    if not AZURE_KEY or not AZURE_REGION:
        logger.error("Azure credentials not configured")
        return None

    try:
        speech_config = speechsdk.SpeechConfig(subscription=AZURE_KEY, region=AZURE_REGION)
        speech_config.speech_recognition_language = "en-US"

        pron_config = speechsdk.PronunciationAssessmentConfig(
            reference_text=reference_text,
            grading_system=speechsdk.PronunciationAssessmentGradingSystem.HundredMark,
            granularity=speechsdk.PronunciationAssessmentGranularity.Phoneme,
            enable_miscue=True,
        )

        audio_config = speechsdk.audio.AudioConfig(filename=str(audio_path))
        recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config, audio_config=audio_config
        )
        pron_config.apply_to(recognizer)

        result = recognizer.recognize_once()

        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            pa_result = speechsdk.PronunciationAssessmentResult(result)
            return {
                "pron_score": pa_result.pronunciation_score,
                "accuracy_score": pa_result.accuracy_score,
                "fluency_score": pa_result.fluency_score,
                "completeness_score": pa_result.completeness_score,
                "recognised_text": result.text,
            }
        else:
            logger.warning(
                "Azure failed for '%s': reason=%s", reference_text, result.reason
            )
            if result.reason == speechsdk.ResultReason.Canceled:
                cancel = result.cancellation_details
                logger.warning("  Cancel reason: %s — %s", cancel.reason, cancel.error_details)
            return None
    except Exception as e:
        logger.error("Azure error for '%s': %s", reference_text, e)
        return None


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(df: pd.DataFrame, output_path: str):
    """Generate scatter plots comparing Azure vs SpeakRight scores."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    score_types = [
        ("pron", "Pronunciation"),
        ("acc", "Accuracy"),
        ("fluency", "Fluency"),
    ]

    for ax, (key, label) in zip(axes, score_types):
        azure_col = f"azure_{key}"
        sr_col = f"speakright_{key}"

        if azure_col not in df.columns or sr_col not in df.columns:
            continue

        ax.scatter(
            df[azure_col],
            df[sr_col],
            alpha=0.7,
            edgecolors="k",
            linewidths=0.5,
            s=60,
            c="#4A90D9",
        )

        # Perfect agreement line
        lims = [0, 105]
        ax.plot(lims, lims, "r--", linewidth=1, alpha=0.7, label="Perfect agreement")

        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel(f"Azure {label} Score", fontsize=11)
        ax.set_ylabel(f"SpeakRight {label} Score", fontsize=11)
        ax.set_title(f"{label} Score (n={len(df)})", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

        # Add correlation text
        if len(df) > 2:
            corr = np.corrcoef(df[azure_col], df[sr_col])[0, 1]
            mae = np.mean(np.abs(df[azure_col] - df[sr_col]))
            ax.text(
                0.05, 0.95,
                f"r = {corr:.3f}\nMAE = {mae:.1f}",
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )

    # Add word labels to first plot
    if len(df) <= 30:
        ax0 = axes[0]
        for _, row in df.iterrows():
            ax0.annotate(
                row["text"][:12],
                (row["azure_pron"], row["speakright_pron"]),
                fontsize=7,
                alpha=0.6,
                xytext=(3, 3),
                textcoords="offset points",
            )

    plt.suptitle(
        "SpeakRight (wav2vec2 + GOP) vs Azure Pronunciation Assessment",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    logger.info("Plot saved to %s", output_path)


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def main():
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_texts = WORD_LIST + SENTENCE_LIST
    text_types = ["word"] * len(WORD_LIST) + ["sentence"] * len(SENTENCE_LIST)

    # Step 1: Generate TTS audio for all test items
    logger.info("=== Step 1: Generating %d TTS audio files ===", len(all_texts))
    audio_paths = {}
    for text in all_texts:
        safe_name = text.lower().replace(" ", "_")[:40]
        audio_path = AUDIO_DIR / f"{safe_name}.wav"
        if audio_path.exists():
            logger.info("  [cached] %s", text)
            audio_paths[text] = audio_path
        else:
            if generate_tts_audio(text, audio_path):
                logger.info("  [generated] %s", text)
                audio_paths[text] = audio_path
                time.sleep(0.5)  # Rate limit gTTS
            else:
                logger.warning("  [FAILED] %s", text)

    logger.info("Generated %d / %d audio files", len(audio_paths), len(all_texts))

    # Step 2: Load SpeakRight scorer
    logger.info("\n=== Step 2: Loading SpeakRight model ===")
    sys.path.insert(0, ".")
    from src.models.wav2vec2_scorer import Wav2Vec2PronunciationScorer
    scorer = Wav2Vec2PronunciationScorer(model_name="facebook/wav2vec2-base")
    logger.info("SpeakRight model loaded on %s", scorer.device)

    # Step 3: Score each audio with both systems
    logger.info("\n=== Step 3: Running benchmark (%d samples) ===", len(audio_paths))
    records = []

    for i, (text, ttype) in enumerate(zip(all_texts, text_types)):
        if text not in audio_paths:
            continue

        audio_path = audio_paths[text]
        logger.info("[%d/%d] '%s'...", i + 1, len(audio_paths), text[:40])

        # SpeakRight
        sr = score_with_speakright(audio_path, text, scorer)
        if sr is None:
            logger.warning("  SpeakRight failed — skipping")
            continue

        # Azure
        az = score_with_azure(audio_path, text)
        if az is None:
            logger.warning("  Azure failed — recording SpeakRight-only result")
            records.append({
                "text": text,
                "type": ttype,
                "speakright_pron": sr["pron_score"],
                "speakright_acc": sr["accuracy_score"],
                "speakright_fluency": sr["fluency_score"],
                "speakright_completeness": sr["completeness_score"],
                "speakright_recognised": sr["recognised_text"],
                "azure_pron": None,
                "azure_acc": None,
                "azure_fluency": None,
                "azure_completeness": None,
                "azure_recognised": None,
            })
            continue

        record = {
            "text": text,
            "type": ttype,
            "speakright_pron": sr["pron_score"],
            "azure_pron": az["pron_score"],
            "speakright_acc": sr["accuracy_score"],
            "azure_acc": az["accuracy_score"],
            "speakright_fluency": sr["fluency_score"],
            "azure_fluency": az["fluency_score"],
            "speakright_completeness": sr["completeness_score"],
            "azure_completeness": az["completeness_score"],
            "speakright_recognised": sr["recognised_text"],
            "azure_recognised": az["recognised_text"],
        }
        records.append(record)

        logger.info(
            "  SR: pron=%.1f acc=%.1f | AZ: pron=%.1f acc=%.1f",
            sr["pron_score"], sr["accuracy_score"],
            az["pron_score"], az["accuracy_score"],
        )

    # Step 4: Save results
    logger.info("\n=== Step 4: Saving results ===")
    df = pd.DataFrame(records)
    csv_path = RESULTS_DIR / "benchmark.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Saved %d rows to %s", len(df), csv_path)

    # Print summary
    df_both = df.dropna(subset=["azure_pron"])
    if not df_both.empty:
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        print(f"Total samples:     {len(df)}")
        print(f"Both scored:       {len(df_both)}")
        print(f"SpeakRight only:   {len(df) - len(df_both)}")
        print()
        print("Score Averages:")
        print(f"  SpeakRight PronScore:  {df_both['speakright_pron'].mean():.1f} (±{df_both['speakright_pron'].std():.1f})")
        print(f"  Azure PronScore:       {df_both['azure_pron'].mean():.1f} (±{df_both['azure_pron'].std():.1f})")
        print(f"  SpeakRight Accuracy:   {df_both['speakright_acc'].mean():.1f}")
        print(f"  Azure Accuracy:        {df_both['azure_acc'].mean():.1f}")
        print()
        print("Correlation (Pearson r):")
        if len(df_both) > 2:
            for metric in ["pron", "acc", "fluency", "completeness"]:
                sr_col = f"speakright_{metric}"
                az_col = f"azure_{metric}"
                corr = np.corrcoef(df_both[sr_col], df_both[az_col])[0, 1]
                mae = np.mean(np.abs(df_both[sr_col] - df_both[az_col]))
                print(f"  {metric:15s}  r={corr:.3f}  MAE={mae:.1f}")
        print("=" * 60)

        # Step 5: Generate plot
        logger.info("\n=== Step 5: Generating comparison plot ===")
        plot_path = RESULTS_DIR / "comparison_plot.png"
        plot_comparison(df_both, str(plot_path))
    else:
        logger.warning("No samples scored by both systems — cannot generate comparison plot")
        # Generate SpeakRight-only plot if we have data
        if not df.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            scores = df["speakright_pron"].values
            texts = df["text"].values
            bars = ax.barh(range(len(scores)), scores, color="#4A90D9", edgecolor="k", linewidth=0.5)
            ax.set_yticks(range(len(scores)))
            ax.set_yticklabels([t[:25] for t in texts], fontsize=9)
            ax.set_xlabel("SpeakRight Pronunciation Score", fontsize=11)
            ax.set_title("SpeakRight Pronunciation Assessment (wav2vec2 + GOP)", fontsize=13, fontweight="bold")
            ax.set_xlim(0, 105)
            ax.grid(axis="x", alpha=0.3)
            for bar, score in zip(bars, scores):
                ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                        f"{score:.1f}", va="center", fontsize=8)
            plt.tight_layout()
            plot_path = RESULTS_DIR / "comparison_plot.png"
            plt.savefig(str(plot_path), dpi=150, bbox_inches="tight")
            logger.info("SpeakRight-only plot saved to %s", plot_path)

    logger.info("\nBenchmark complete!")


if __name__ == "__main__":
    main()
