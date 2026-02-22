"""
Run the full SpeakRight vs Azure benchmark and produce a comparison report.

Usage:
    python scripts/benchmark_azure.py \
        --words data/test_words.txt \
        --audio-dir data/recordings/ \
        --output results/benchmark.csv \
        --plot results/comparison_plot.png

Outputs:
    - CSV with per-word scores from both systems
    - Pearson r, MAE, RMSE between systems
    - Optional matplotlib scatter plot
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.evaluation.azure_compare import run_benchmark
from src.evaluation.metrics import summarise_comparison

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def plot_comparison(df: pd.DataFrame, output_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, score in zip(axes, ["pron", "acc"]):
        ax.scatter(
            df[f"azure_{score}"],
            df[f"speakright_{score}"],
            alpha=0.6,
            edgecolors="k",
            linewidths=0.5,
        )
        lims = [0, 100]
        ax.plot(lims, lims, "r--", linewidth=1, label="Perfect agreement")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel(f"Azure {score.capitalize()}Score")
        ax.set_ylabel(f"SpeakRight {score.capitalize()}Score")
        ax.set_title(f"{score.capitalize()} Score Comparison (n={len(df)})")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    logger.info("Plot saved to %s", output_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--words", required=True, help="Text file: one word per line")
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--output", default="results/benchmark.csv")
    parser.add_argument("--plot", default="results/comparison_plot.png")
    parser.add_argument("--model", default="facebook/wav2vec2-base")
    args = parser.parse_args()

    word_list = Path(args.words).read_text().strip().splitlines()
    logger.info("Benchmarking %d words …", len(word_list))

    df = run_benchmark(
        audio_dir=args.audio_dir,
        word_list=word_list,
        output_csv=args.output,
        speakright_model=args.model,
    )

    if df.empty:
        logger.error("No results produced. Check audio files and Azure credentials.")
        return

    records = df.to_dict("records")
    summary = summarise_comparison(records)
    print("\n=== Benchmark Summary ===")
    print(json.dumps(summary, indent=2))

    Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
    plot_comparison(df, args.plot)


if __name__ == "__main__":
    main()
