#!/usr/bin/env python3
"""Generate consolidated benchmark metrics and plots.

Reads:
  - results/benchmark.csv
  - results/benchmark_diverse.csv

Writes:
  - results/evaluation_summary.json
  - results/evaluation_summary.png
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.use("Agg")


QUALITY_ORDER = ["native", "good", "medium", "poor", "wrong", "misread"]
TIER_ORDER = ["native", "good", "medium", "poor", "wrong"]
MISPRON_QUALITY = {"poor", "wrong", "misread"}


def score_to_tier(score: float) -> str:
    if score >= 90:
        return "native"
    if score >= 80:
        return "good"
    if score >= 60:
        return "medium"
    if score >= 40:
        return "poor"
    return "wrong"


def expected_tier_from_quality(quality: str) -> str:
    quality = str(quality).lower()
    if quality in {"native", "good", "medium", "poor", "wrong"}:
        return quality
    if quality == "misread":
        return "wrong"
    return "wrong"


def regression_metrics(df: pd.DataFrame, sr_col: str, az_col: str) -> dict[str, float]:
    pair = df[[sr_col, az_col]].dropna()
    if pair.empty:
        return {"n": 0, "pearson_r": float("nan"), "mae": float("nan"), "rmse": float("nan")}

    diff = pair[sr_col] - pair[az_col]
    return {
        "n": int(len(pair)),
        "pearson_r": float(pair[sr_col].corr(pair[az_col])),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(np.square(diff)))),
    }


def binary_f1(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    results_dir = root / "results"
    benchmark_path = results_dir / "benchmark.csv"
    diverse_path = results_dir / "benchmark_diverse.csv"

    if not benchmark_path.exists() or not diverse_path.exists():
        raise FileNotFoundError(
            "Missing required CSV files. Expected results/benchmark.csv and results/benchmark_diverse.csv"
        )

    benchmark = pd.read_csv(benchmark_path)
    diverse = pd.read_csv(diverse_path)

    benchmark_metrics = regression_metrics(benchmark, "speakright_pron", "azure_pron")
    diverse_metrics = regression_metrics(diverse, "speakright_pron", "azure_pron")

    combined = pd.concat(
        [
            benchmark[["speakright_pron", "azure_pron"]],
            diverse[["speakright_pron", "azure_pron"]],
        ],
        ignore_index=True,
    )
    overall_metrics = regression_metrics(combined, "speakright_pron", "azure_pron")

    diverse_eval = diverse.copy()
    diverse_eval["quality"] = diverse_eval["quality"].str.lower()
    diverse_eval["expected_tier"] = diverse_eval["quality"].map(expected_tier_from_quality)
    diverse_eval["pred_tier"] = diverse_eval["speakright_pron"].map(score_to_tier)
    diverse_eval["tier_match"] = diverse_eval["expected_tier"] == diverse_eval["pred_tier"]

    per_tier_accuracy = (
        diverse_eval.groupby("quality")["tier_match"]
        .mean()
        .reindex(QUALITY_ORDER)
        .dropna()
        .to_dict()
    )
    overall_tier_accuracy = float(diverse_eval["tier_match"].mean())

    y_true = diverse_eval["quality"].isin(MISPRON_QUALITY).astype(int).to_numpy()
    y_pred = (diverse_eval["speakright_pron"] < 60.0).astype(int).to_numpy()
    mispron_f1 = binary_f1(y_true, y_pred)

    summary_table = [
        {"metric": "Pearson r (benchmark)", "value": round(benchmark_metrics["pearson_r"], 4)},
        {"metric": "MAE (benchmark)", "value": round(benchmark_metrics["mae"], 4)},
        {"metric": "RMSE (benchmark)", "value": round(benchmark_metrics["rmse"], 4)},
        {"metric": "Pearson r (diverse)", "value": round(diverse_metrics["pearson_r"], 4)},
        {"metric": "MAE (diverse)", "value": round(diverse_metrics["mae"], 4)},
        {"metric": "RMSE (diverse)", "value": round(diverse_metrics["rmse"], 4)},
        {"metric": "Pearson r (overall)", "value": round(overall_metrics["pearson_r"], 4)},
        {"metric": "MAE (overall)", "value": round(overall_metrics["mae"], 4)},
        {"metric": "RMSE (overall)", "value": round(overall_metrics["rmse"], 4)},
        {"metric": "Tier accuracy (overall)", "value": round(overall_tier_accuracy, 4)},
        {"metric": "Mispronunciation F1", "value": round(mispron_f1["f1"], 4)},
    ]

    for quality in QUALITY_ORDER:
        if quality in per_tier_accuracy:
            summary_table.append(
                {
                    "metric": f"Tier accuracy ({quality})",
                    "value": round(float(per_tier_accuracy[quality]), 4),
                }
            )

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "benchmark": str(benchmark_path.relative_to(root)),
            "benchmark_diverse": str(diverse_path.relative_to(root)),
        },
        "thresholds": {
            "tier_mapping": {
                "native": ">=90",
                "good": "80-89.99",
                "medium": "60-79.99",
                "poor": "40-59.99",
                "wrong": "<40",
            },
            "mispronunciation_threshold": "speakright_pron < 60",
            "mispronunciation_positive_labels": sorted(MISPRON_QUALITY),
        },
        "regression": {
            "benchmark": benchmark_metrics,
            "diverse": diverse_metrics,
            "overall": overall_metrics,
        },
        "tier_accuracy": {
            "overall": overall_tier_accuracy,
            "per_quality": per_tier_accuracy,
        },
        "mispronunciation_detection": mispron_f1,
        "summary_table": summary_table,
    }

    summary_json_path = results_dir / "evaluation_summary.json"
    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.scatter(
        benchmark["azure_pron"],
        benchmark["speakright_pron"],
        c="#1f77b4",
        alpha=0.8,
        edgecolors="white",
    )
    line_min = float(min(benchmark["azure_pron"].min(), benchmark["speakright_pron"].min()))
    line_max = float(max(benchmark["azure_pron"].max(), benchmark["speakright_pron"].max()))
    ax.plot([line_min, line_max], [line_min, line_max], "k--", linewidth=1)
    ax.set_title("Benchmark: SpeakRight vs Azure")
    ax.set_xlabel("Azure PronScore")
    ax.set_ylabel("SpeakRight PronScore")
    ax.text(
        0.02,
        0.98,
        f"r={benchmark_metrics['pearson_r']:.3f}\nMAE={benchmark_metrics['mae']:.2f}\nRMSE={benchmark_metrics['rmse']:.2f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )

    ax = axes[0, 1]
    grouped = (
        diverse.groupby("quality")[["speakright_pron", "azure_pron"]]
        .mean()
        .reindex([q for q in QUALITY_ORDER if q in set(diverse["quality"].str.lower())])
    )
    x = np.arange(len(grouped.index))
    width = 0.36
    ax.bar(x - width / 2, grouped["speakright_pron"], width=width, label="SpeakRight", color="#2ca02c")
    ax.bar(x + width / 2, grouped["azure_pron"], width=width, label="Azure", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(grouped.index, rotation=25)
    ax.set_ylim(0, 105)
    ax.set_title("Mean PronScore by Quality Tier")
    ax.set_ylabel("PronScore")
    ax.legend()

    ax = axes[1, 0]
    per_tier_items = [(k, per_tier_accuracy[k]) for k in QUALITY_ORDER if k in per_tier_accuracy]
    labels = [k for k, _ in per_tier_items]
    values = [v for _, v in per_tier_items]
    ax.bar(labels, values, color="#9467bd")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-tier Accuracy (SpeakRight)")
    ax.set_ylabel("Accuracy")
    ax.tick_params(axis="x", rotation=25)
    for i, value in enumerate(values):
        ax.text(i, value + 0.02, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    ax = axes[1, 1]
    cm = pd.crosstab(
        pd.Categorical(diverse_eval["expected_tier"], categories=TIER_ORDER),
        pd.Categorical(diverse_eval["pred_tier"], categories=TIER_ORDER),
    )
    im = ax.imshow(cm.values, cmap="Blues")
    ax.set_title("Tier Confusion Matrix")
    ax.set_xlabel("Predicted tier")
    ax.set_ylabel("Expected tier")
    ax.set_xticks(np.arange(len(TIER_ORDER)))
    ax.set_xticklabels(TIER_ORDER, rotation=30)
    ax.set_yticks(np.arange(len(TIER_ORDER)))
    ax.set_yticklabels(TIER_ORDER)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(int(cm.iat[i, j])), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Full Evaluation Summary | Mispronunciation F1={mispron_f1['f1']:.3f}",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    summary_png_path = results_dir / "evaluation_summary.png"
    fig.savefig(summary_png_path, dpi=200)
    plt.close(fig)

    print(f"Saved {summary_json_path}")
    print(f"Saved {summary_png_path}")


if __name__ == "__main__":
    main()
