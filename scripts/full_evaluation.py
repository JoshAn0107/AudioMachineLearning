#!/usr/bin/env python3
# pyright: reportMissingTypeStubs=false, reportAttributeAccessIssue=false, reportUnknownVariableType=false, reportUnknownMemberType=false, reportArgumentType=false, reportUnknownArgumentType=false, reportAny=false, reportUnusedCallResult=false
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
from matplotlib.patches import Rectangle
from scipy.stats import kendalltau, spearmanr

matplotlib.use("Agg")


QUALITY_ORDER = ["native", "good", "medium", "poor", "wrong", "misread"]
TIER_ORDER = ["native", "good", "medium", "poor", "wrong"]
MISPRON_QUALITY = {"poor", "wrong", "misread"}
ACCEPTABLE_QUALITY = {"native", "good", "medium"}
PROBLEMATIC_QUALITY = {"poor", "wrong", "misread"}
RANK = {"native": 0, "good": 1, "medium": 2, "poor": 3, "wrong": 4, "misread": 4}


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
    diverse_eval["expected_rank"] = diverse_eval["quality"].map(RANK)
    diverse_eval["pred_rank"] = diverse_eval["pred_tier"].map(RANK)
    diverse_eval["tier_match"] = diverse_eval["expected_tier"] == diverse_eval["pred_tier"]
    diverse_eval["tier_within_one"] = (diverse_eval["expected_rank"] - diverse_eval["pred_rank"]).abs() <= 1

    diverse_eval["expected_binary_problematic"] = diverse_eval["quality"].isin(PROBLEMATIC_QUALITY)
    diverse_eval["pred_binary_problematic"] = diverse_eval["speakright_pron"] < 60.0

    per_tier_accuracy = (
        diverse_eval.groupby("quality")["tier_match"]
        .mean()
        .reindex(QUALITY_ORDER)
        .dropna()
        .to_dict()
    )
    tier_accuracy_strict = float(diverse_eval["tier_match"].mean())
    tier_accuracy_within_one = float(diverse_eval["tier_within_one"].mean())
    binary_acceptable_accuracy = float(
        (diverse_eval["expected_binary_problematic"] == diverse_eval["pred_binary_problematic"]).mean()
    )

    corr_df = diverse_eval[["expected_rank", "speakright_pron"]].dropna()
    spearman_r, spearman_p = spearmanr(corr_df["expected_rank"], -corr_df["speakright_pron"])
    kendall_tau, kendall_p = kendalltau(corr_df["expected_rank"], -corr_df["speakright_pron"])
    ordinal_correlation = {
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
        "kendall_tau": float(kendall_tau),
        "kendall_p": float(kendall_p),
    }

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
        {"metric": "Tier accuracy strict (diverse)", "value": round(tier_accuracy_strict, 4)},
        {"metric": "Tier accuracy within one (diverse)", "value": round(tier_accuracy_within_one, 4)},
        {"metric": "Binary acceptable accuracy (diverse)", "value": round(binary_acceptable_accuracy, 4)},
        {"metric": "Spearman rho (diverse tiers)", "value": round(ordinal_correlation["spearman_r"], 4)},
        {"metric": "Kendall tau (diverse tiers)", "value": round(ordinal_correlation["kendall_tau"], 4)},
        {"metric": "Mispronunciation F1", "value": round(mispron_f1["f1"], 4)},
    ]

    for quality in QUALITY_ORDER:
        if quality in per_tier_accuracy:
            summary_table.append(
                {
                    "metric": f"Tier accuracy strict ({quality})",
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
            "tier_rank_mapping": RANK,
            "binary_acceptable_labels": sorted(ACCEPTABLE_QUALITY),
            "binary_problematic_labels": sorted(PROBLEMATIC_QUALITY),
            "binary_problematic_threshold": "speakright_pron < 60",
            "mispronunciation_threshold": "speakright_pron < 60",
            "mispronunciation_positive_labels": sorted(MISPRON_QUALITY),
        },
        "regression": {
            "benchmark": benchmark_metrics,
            "diverse": diverse_metrics,
            "overall": overall_metrics,
        },
        "tier_metrics": {
            "tier_accuracy_strict": tier_accuracy_strict,
            "tier_accuracy_within_one": tier_accuracy_within_one,
            "binary_acceptable_accuracy": binary_acceptable_accuracy,
            "ordinal_correlation": ordinal_correlation,
        },
        "tier_accuracy": {
            "overall": tier_accuracy_strict,
            "strict": tier_accuracy_strict,
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
        diverse_eval.groupby("quality")[["speakright_pron", "azure_pron"]]
        .mean()
        .reindex([q for q in QUALITY_ORDER if q in set(diverse_eval["quality"])])
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
    metric_labels = [
        "Strict accuracy",
        "Within-one accuracy",
        "Binary acceptable",
        "Spearman rho",
        "Kendall tau",
    ]
    metric_values = [
        tier_accuracy_strict,
        tier_accuracy_within_one,
        binary_acceptable_accuracy,
        ordinal_correlation["spearman_r"],
        ordinal_correlation["kendall_tau"],
    ]
    metric_colors = ["#9467bd", "#17becf", "#8c564b", "#2ca02c", "#bcbd22"]
    ypos = np.arange(len(metric_labels))
    ax.barh(ypos, metric_values, color=metric_colors)
    ax.set_yticks(ypos)
    ax.set_yticklabels(metric_labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Value")
    ax.set_title("Tier-related Metrics (diverse set)")
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    for i, value in enumerate(metric_values):
        ax.text(min(value + 0.02, 1.01), i, f"{value:.3f}", va="center", ha="left", fontsize=9)
    ax.text(
        0.98,
        0.02,
        f"p(rho)={ordinal_correlation['spearman_p']:.2e}\np(tau)={ordinal_correlation['kendall_p']:.2e}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )

    ax = axes[1, 1]
    cm = pd.crosstab(
        pd.Categorical(diverse_eval["expected_tier"], categories=TIER_ORDER),
        pd.Categorical(diverse_eval["pred_tier"], categories=TIER_ORDER),
    )
    im = ax.imshow(cm.values, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if abs(i - j) <= 1:
                ax.add_patch(
                    Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        facecolor="#f1c40f",
                        edgecolor="#f39c12",
                        linewidth=0.8,
                        alpha=0.12,
                    )
                )
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
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    summary_png_path = results_dir / "evaluation_summary.png"
    fig.savefig(summary_png_path, dpi=200)
    plt.close(fig)

    print(f"Saved {summary_json_path}")
    print(f"Saved {summary_png_path}")


if __name__ == "__main__":
    main()
