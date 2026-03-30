#!/usr/bin/env python3
"""
Feature Extraction Comparison: MFCC vs Mel-Spectrogram vs Wav2Vec2 Embeddings.

Compares three feature representations for pronunciation quality classification:
  1. MFCC (13 coefficients + delta + delta-delta = 39 features)
  2. Mel-Spectrogram (128 mel bands, mean-pooled)
  3. Wav2Vec2 contextual embeddings (768-dim for base, 1024 for large)

Each is used to train an SVM and Random Forest classifier to predict
audio quality tier (native/good/medium/poor/wrong).

Outputs:
  - results/feature_comparison.csv  (per-classifier accuracy)
  - results/feature_comparison.png  (bar chart)
"""

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DIVERSE_DIR = Path("data/diverse_audio")
RESULTS_DIR = Path("results")

# Quality tiers and their expected labels
QUALITY_TIERS = ["native", "good", "medium", "poor", "wrong"]

# Words that have all 5 quality variants
WORDS = ["hello", "beautiful", "technology", "apple", "environment",
         "congratulations", "university", "responsibility"]
SENTENCES = [
    "The quick brown fox jumps over the lazy dog",
    "She sells seashells by the seashore",
    "I would like a cup of coffee please",
]


def extract_mfcc(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Extract 39-dim MFCC features (13 + delta + delta-delta), mean-pooled."""
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)
    features = np.concatenate([mfcc, delta, delta2], axis=0)  # (39, T)
    return np.concatenate([features.mean(axis=1), features.std(axis=1)])  # (78,)


def extract_mel_spectrogram(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Extract 128-band log-mel spectrogram, mean-pooled."""
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return np.concatenate([log_mel.mean(axis=1), log_mel.std(axis=1)])  # (256,)


def extract_wav2vec2(audio: np.ndarray, sr: int, processor, model) -> np.ndarray:
    """Extract wav2vec2 contextual embeddings, mean-pooled."""
    inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(inputs.input_values, output_hidden_states=True)
    # Use last hidden state (before CTC head)
    hidden = outputs.hidden_states[-1].squeeze(0).numpy()  # (T, 1024)
    return np.concatenate([hidden.mean(axis=0), hidden.std(axis=0)])  # (2048,)


def load_dataset():
    """Load diverse audio files with quality labels."""
    all_texts = WORDS + SENTENCES
    X_audio = []
    y_labels = []
    file_paths = []

    for text in all_texts:
        safe_name = text.lower().replace(" ", "_")[:40]
        for quality in QUALITY_TIERS:
            path = DIVERSE_DIR / f"{safe_name}_{quality}.wav"
            if path.exists():
                audio, sr = sf.read(str(path), dtype="float32")
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                X_audio.append((audio, sr))
                y_labels.append(quality)
                file_paths.append(str(path))

    logger.info("Loaded %d samples across %d quality tiers", len(X_audio), len(set(y_labels)))
    return X_audio, y_labels, file_paths


def run_comparison():
    """Run feature extraction and classification comparison."""
    X_audio, y_labels, _ = load_dataset()
    le = LabelEncoder()
    y = le.fit_transform(y_labels)

    # Extract features for each method
    logger.info("Extracting MFCC features...")
    X_mfcc = np.array([extract_mfcc(a, sr) for a, sr in X_audio])
    logger.info("  Shape: %s", X_mfcc.shape)

    logger.info("Extracting Mel-Spectrogram features...")
    X_mel = np.array([extract_mel_spectrogram(a, sr) for a, sr in X_audio])
    logger.info("  Shape: %s", X_mel.shape)

    logger.info("Loading wav2vec2 for embedding extraction...")
    model_name = os.getenv("SPEAKRIGHT_MODEL", "facebook/wav2vec2-large-960h")
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    model.eval()

    logger.info("Extracting Wav2Vec2 embeddings...")
    X_w2v = np.array([extract_wav2vec2(a, sr, processor, model) for a, sr in X_audio])
    logger.info("  Shape: %s", X_w2v.shape)

    # Free model memory
    del model, processor
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Classifiers to test
    classifiers = {
        "SVM (RBF)": SVC(kernel="rbf", C=10, gamma="scale"),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }

    feature_sets = {
        "MFCC (78-dim)": X_mfcc,
        "Mel-Spec (256-dim)": X_mel,
        "Wav2Vec2 (2048-dim)": X_w2v,
    }

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = []

    for feat_name, X in feature_sets.items():
        for clf_name, clf in classifiers.items():
            pipeline = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", clf),
            ])
            scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
            mean_acc = scores.mean()
            std_acc = scores.std()
            results.append({
                "features": feat_name,
                "classifier": clf_name,
                "accuracy_mean": round(mean_acc * 100, 1),
                "accuracy_std": round(std_acc * 100, 1),
                "cv_scores": [round(s * 100, 1) for s in scores],
            })
            logger.info("  %s + %s: %.1f%% (±%.1f%%)", feat_name, clf_name, mean_acc * 100, std_acc * 100)

    return results, feature_sets, y, le


def plot_results(results):
    """Generate grouped bar chart of classification results."""
    fig, ax = plt.subplots(figsize=(12, 6))

    features = sorted(set(r["features"] for r in results))
    classifiers = sorted(set(r["classifier"] for r in results))
    x = np.arange(len(features))
    width = 0.35
    colors = ["#4A90D9", "#E8A838"]

    for i, clf_name in enumerate(classifiers):
        accs = []
        errs = []
        for feat in features:
            r = next(r for r in results if r["features"] == feat and r["classifier"] == clf_name)
            accs.append(r["accuracy_mean"])
            errs.append(r["accuracy_std"])
        bars = ax.bar(x + i * width - width / 2, accs, width, yerr=errs,
                      label=clf_name, color=colors[i], edgecolor="k", linewidth=0.5,
                      capsize=3)
        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 2,
                    f"{acc:.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Classification Accuracy (%)", fontsize=12)
    ax.set_title("Feature Comparison: Audio Quality Classification (5-class, 5-fold CV)",
                 fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(features, fontsize=11)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = RESULTS_DIR / "feature_comparison.png"
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    logger.info("Plot saved to %s", out_path)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results, feature_sets, y, le = run_comparison()

    # Save results
    import pandas as pd
    df = pd.DataFrame(results)
    csv_path = RESULTS_DIR / "feature_comparison.csv"
    df.to_csv(csv_path, index=False)
    logger.info("Results saved to %s", csv_path)

    # Print summary
    print("\n" + "=" * 70)
    print("FEATURE COMPARISON RESULTS (5-class quality classification)")
    print("=" * 70)
    for r in sorted(results, key=lambda x: -x["accuracy_mean"]):
        print(f"  {r['features']:25s} + {r['classifier']:15s}  {r['accuracy_mean']:5.1f}% (±{r['accuracy_std']:.1f}%)")
    print("=" * 70)

    plot_results(results)
    logger.info("Feature comparison complete!")


if __name__ == "__main__":
    main()
