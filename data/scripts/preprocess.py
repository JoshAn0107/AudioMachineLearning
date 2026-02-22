"""
Preprocess raw datasets into a unified format for training and evaluation.

Steps:
  1. Resample audio to 16 kHz mono
  2. Remove clips shorter than 0.1 s or longer than 30 s
  3. Normalise transcript text
  4. (Optional) Run Montreal Forced Aligner to produce phoneme alignments
  5. Save as HuggingFace Dataset to data/processed/

Usage:
    python data/scripts/preprocess.py --source librispeech --split train_clean_100
"""

import argparse
import logging
import re
from pathlib import Path

import numpy as np
from datasets import Audio, Dataset, load_from_disk

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TARGET_SR = 16_000
MIN_DURATION_SEC = 0.1
MAX_DURATION_SEC = 30.0


def normalise_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z\s']", "", text)  # keep letters, spaces, apostrophes
    text = re.sub(r"\s+", " ", text)
    return text


def filter_duration(batch):
    dur = len(batch["audio"]["array"]) / batch["audio"]["sampling_rate"]
    return MIN_DURATION_SEC <= dur <= MAX_DURATION_SEC


def preprocess_batch(batch):
    batch["text"] = normalise_text(batch["text"])
    # Normalise audio amplitude to [-1, 1]
    audio = np.array(batch["audio"]["array"], dtype=np.float32)
    max_amp = np.abs(audio).max()
    if max_amp > 0:
        audio = audio / max_amp
    batch["audio"]["array"] = audio
    return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, choices=["timit", "librispeech", "vctk", "common_voice"])
    parser.add_argument("--split", default="train_clean_100")
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--num-proc", type=int, default=4)
    args = parser.parse_args()

    raw_path = Path(args.raw_dir) / args.source / args.split
    out_path = Path(args.output_dir) / args.source / args.split

    logger.info("Loading dataset from %s …", raw_path)
    ds: Dataset = load_from_disk(str(raw_path))

    logger.info("Original size: %d samples", len(ds))

    ds = ds.cast_column("audio", Audio(sampling_rate=TARGET_SR))
    ds = ds.filter(filter_duration, num_proc=args.num_proc)
    ds = ds.map(preprocess_batch, num_proc=args.num_proc)

    logger.info("Filtered size: %d samples", len(ds))

    out_path.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out_path))
    logger.info("Saved preprocessed dataset to %s", out_path)


if __name__ == "__main__":
    main()
