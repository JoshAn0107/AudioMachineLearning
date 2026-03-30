#!/usr/bin/env python3
"""
Grid search for GOP sigmoid calibration parameters (alpha, beta).
Uses diverse benchmark audio with Azure scores as ground truth.

Minimizes MAE(SpeakRight_accuracy, Azure_accuracy) across all samples.
"""

import json
import logging
import os
import sys
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import soundfile as sf
import torch
import torch.nn.functional as F
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")


def load_azure_ground_truth():
    csv_path = RESULTS_DIR / "benchmark_diverse.csv"
    if not csv_path.exists():
        csv_path = RESULTS_DIR / "benchmark.csv"
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["azure_acc", "azure_pron"])
    return df


def compute_gop_scores_raw(audio_path, reference_text, processor, model, vocab, blank_id):
    audio, sr = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values).logits

    log_probs = F.log_softmax(logits, dim=-1).squeeze(0).numpy()
    predicted_ids = logits.argmax(dim=-1).squeeze(0).tolist()

    aligned_log_probs = []
    prev = None
    for frame_idx, tid in enumerate(predicted_ids):
        if tid == blank_id or tid == prev:
            prev = tid
            continue
        char = vocab.get(tid, "")
        if char != "|":
            aligned_log_probs.append(log_probs[frame_idx, tid])
        prev = tid

    return aligned_log_probs


def sigmoid_score(log_prob, alpha, beta):
    return 100.0 / (1.0 + np.exp(-(alpha * log_prob + beta)))


def evaluate_calibration(all_raw_scores, azure_accs, alpha, beta):
    pred_accs = []
    for raw_scores in all_raw_scores:
        if not raw_scores:
            pred_accs.append(0.0)
            continue
        char_scores = [sigmoid_score(lp, alpha, beta) for lp in raw_scores]
        pred_accs.append(np.mean(char_scores))

    pred_accs = np.array(pred_accs)
    azure_accs = np.array(azure_accs)
    mae = np.mean(np.abs(pred_accs - azure_accs))
    return mae, pred_accs


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = load_azure_ground_truth()
    logger.info("Loaded %d samples with Azure ground truth", len(df))

    model_name = os.getenv("SPEAKRIGHT_MODEL", "facebook/wav2vec2-large-960h")
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    model.eval()
    vocab = {v: k for k, v in processor.tokenizer.get_vocab().items()}
    blank_id = processor.tokenizer.pad_token_id

    logger.info("Extracting raw GOP log-probs for all samples...")

    all_raw = []
    azure_accs = []
    azure_prons = []
    valid_indices = []

    for idx, row in df.iterrows():
        text = row["text"]
        safe_name = text.lower().replace(" ", "_")[:40]

        if "quality" in df.columns:
            quality = row["quality"]
            audio_path = Path(f"data/diverse_audio/{safe_name}_{quality}.wav")
        else:
            audio_path = Path(f"data/benchmark_audio/{safe_name}.wav")

        if not audio_path.exists():
            continue

        try:
            raw = compute_gop_scores_raw(audio_path, text, processor, model, vocab, blank_id)
            all_raw.append(raw)
            azure_accs.append(row["azure_acc"])
            azure_prons.append(row["azure_pron"])
            valid_indices.append(idx)
        except Exception as e:
            logger.warning("  Skip %s: %s", audio_path.name, e)

    del model, processor

    logger.info("Got raw scores for %d samples", len(all_raw))

    alpha_range = np.arange(1.0, 8.0, 0.5)
    beta_range = np.arange(0.0, 6.0, 0.5)

    logger.info("Grid search: %d alpha x %d beta = %d combinations",
                len(alpha_range), len(beta_range), len(alpha_range) * len(beta_range))

    best_mae = float("inf")
    best_alpha = 4.0
    best_beta = 3.0
    grid_results = []

    for alpha in alpha_range:
        for beta in beta_range:
            mae, _ = evaluate_calibration(all_raw, azure_accs, alpha, beta)
            grid_results.append({"alpha": alpha, "beta": beta, "mae": mae})
            if mae < best_mae:
                best_mae = mae
                best_alpha = alpha
                best_beta = beta

    logger.info("Best: alpha=%.1f, beta=%.1f, MAE=%.2f", best_alpha, best_beta, best_mae)

    _, best_preds = evaluate_calibration(all_raw, azure_accs, best_alpha, best_beta)
    corr = np.corrcoef(azure_accs, best_preds)[0, 1] if len(azure_accs) > 2 else 0

    print("\n" + "=" * 60)
    print("GOP CALIBRATION GRID SEARCH RESULTS")
    print("=" * 60)
    print(f"  Samples:     {len(all_raw)}")
    print(f"  Best alpha:  {best_alpha:.1f}")
    print(f"  Best beta:   {best_beta:.1f}")
    print(f"  Best MAE:    {best_mae:.2f}")
    print(f"  Pearson r:   {corr:.3f}")
    print(f"  Current:     alpha=4.0, beta=3.0")
    _, current_preds = evaluate_calibration(all_raw, azure_accs, 4.0, 3.0)
    current_mae = np.mean(np.abs(np.array(azure_accs) - current_preds))
    print(f"  Current MAE: {current_mae:.2f}")
    print(f"  Improvement: {current_mae - best_mae:.2f}")
    print("=" * 60)

    grid_df = pd.DataFrame(grid_results)
    grid_df.to_csv(RESULTS_DIR / "calibration_grid.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    pivot = grid_df.pivot(index="beta", columns="alpha", values="mae")
    im = axes[0].imshow(pivot.values, cmap="viridis_r", aspect="auto",
                        extent=[alpha_range[0], alpha_range[-1], beta_range[-1], beta_range[0]])
    axes[0].set_xlabel("Alpha", fontsize=11)
    axes[0].set_ylabel("Beta", fontsize=11)
    axes[0].set_title("Calibration Grid Search (MAE vs Azure)", fontsize=12, fontweight="bold")
    axes[0].plot(best_alpha, best_beta, "r*", markersize=15, label=f"Best: α={best_alpha:.1f}, β={best_beta:.1f}")
    axes[0].legend(fontsize=9)
    plt.colorbar(im, ax=axes[0], label="MAE")

    axes[1].scatter(azure_accs, best_preds, alpha=0.6, edgecolors="k", linewidth=0.5, c="#4A90D9", s=50)
    axes[1].plot([0, 100], [0, 100], "r--", linewidth=1, label="Perfect agreement")
    axes[1].set_xlabel("Azure Accuracy Score", fontsize=11)
    axes[1].set_ylabel("SpeakRight Accuracy Score", fontsize=11)
    axes[1].set_title(f"Optimized Calibration (r={corr:.3f}, MAE={best_mae:.1f})", fontsize=12, fontweight="bold")
    axes[1].set_xlim(0, 105)
    axes[1].set_ylim(0, 105)
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(RESULTS_DIR / "calibration_optimization.png"), dpi=150, bbox_inches="tight")
    logger.info("Plot saved")

    with open(RESULTS_DIR / "best_calibration.json", "w") as f:
        json.dump({"alpha": best_alpha, "beta": best_beta, "mae": best_mae, "pearson_r": corr,
                    "n_samples": len(all_raw), "improvement_over_default": current_mae - best_mae}, f, indent=2)

    logger.info("Calibration optimization complete!")


if __name__ == "__main__":
    main()
