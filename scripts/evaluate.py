"""
Evaluate SpeakRight on a test split and report phoneme/word accuracy metrics.

Usage:
    python scripts/evaluate.py \
        --dataset timit \
        --split test \
        --model facebook/wav2vec2-base
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from datasets import load_from_disk

from src.features.extractor import load_audio
from src.models.wav2vec2_scorer import Wav2Vec2PronunciationScorer
from src.evaluation.metrics import error_detection_metrics

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="timit")
    parser.add_argument("--split", default="test")
    parser.add_argument("--model", default="facebook/wav2vec2-base")
    parser.add_argument("--max-samples", type=int, default=500)
    args = parser.parse_args()

    scorer = Wav2Vec2PronunciationScorer(model_name=args.model)

    ds_path = Path("data/processed") / args.dataset / args.split
    logger.info("Loading %s …", ds_path)
    ds = load_from_disk(str(ds_path))
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    pron_scores = []
    acc_scores = []

    for sample in ds:
        audio = np.array(sample["audio"]["array"], dtype=np.float32)
        sr = sample["audio"]["sampling_rate"]
        text = sample["text"]

        result = scorer.score(audio, text, sample_rate=sr)
        pron_scores.append(result.pron_score)
        acc_scores.append(result.accuracy_score)

    report = {
        "n_samples": len(pron_scores),
        "mean_pron_score": round(float(np.mean(pron_scores)), 2),
        "std_pron_score": round(float(np.std(pron_scores)), 2),
        "mean_accuracy_score": round(float(np.mean(acc_scores)), 2),
    }

    print(json.dumps(report, indent=2))
    Path("results").mkdir(exist_ok=True)
    Path("results/eval_report.json").write_text(json.dumps(report, indent=2))
    logger.info("Report saved to results/eval_report.json")


if __name__ == "__main__":
    main()
