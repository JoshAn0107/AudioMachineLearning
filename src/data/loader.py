"""
Unified dataset loader for TIMIT, LibriSpeech, VCTK, and Common Voice.

All loaders return a HuggingFace Dataset with standardised columns:
    audio       – {"array": np.ndarray, "sampling_rate": int}
    text        – reference transcript (lowercase)
    speaker_id  – speaker identifier string
    dataset     – source dataset name
"""

from __future__ import annotations

import logging
from pathlib import Path

from datasets import Audio, Dataset, DatasetDict, concatenate_datasets, load_dataset

logger = logging.getLogger(__name__)

TARGET_SR = 16_000


def load_timit(local_path: str | Path | None = None) -> DatasetDict:
    """
    Load TIMIT. Requires LDC licence; download manually and set local_path,
    or use the HuggingFace 'timit_asr' dataset (limited split).
    """
    if local_path and Path(local_path).exists():
        ds = load_dataset("timit_asr", data_dir=str(local_path))
    else:
        logger.warning("TIMIT local path not found — loading HF timit_asr (no phoneme labels).")
        ds = load_dataset("timit_asr", trust_remote_code=True)

    def normalise(batch):
        batch["text"] = batch["text"].lower()
        batch["speaker_id"] = batch.get("speaker_id", "unknown")
        batch["dataset"] = "timit"
        return batch

    return ds.cast_column("audio", Audio(sampling_rate=TARGET_SR)).map(normalise)


def load_librispeech(splits: list[str] | None = None) -> DatasetDict:
    """
    Load LibriSpeech from HuggingFace. Splits: train.clean.100, train.clean.360,
    validation.clean, test.clean (and .other variants).
    """
    splits = splits or ["train.clean.100", "validation.clean", "test.clean"]
    datasets = {}
    for split in splits:
        try:
            ds = load_dataset("librispeech_asr", split=split, trust_remote_code=True)
            ds = ds.cast_column("audio", Audio(sampling_rate=TARGET_SR))
            ds = ds.rename_column("text", "text")
            ds = ds.map(lambda b: {"dataset": "librispeech", "text": b["text"].lower()})
            datasets[split.replace(".", "_")] = ds
        except Exception as e:
            logger.error("Failed to load LibriSpeech split '%s': %s", split, e)

    return DatasetDict(datasets)


def load_vctk(local_path: str | Path | None = None) -> Dataset:
    """Load VCTK corpus. 110 speakers, diverse accents."""
    try:
        ds = load_dataset("vctk", trust_remote_code=True)["train"]
    except Exception:
        if local_path:
            ds = load_dataset("audiofolder", data_dir=str(local_path), split="train")
        else:
            raise RuntimeError("Cannot load VCTK. Provide local_path or ensure HF access.")

    ds = ds.cast_column("audio", Audio(sampling_rate=TARGET_SR))
    ds = ds.map(lambda b: {"dataset": "vctk", "text": b.get("text", "").lower()})
    return ds


def load_common_voice(language: str = "en", split: str = "train") -> Dataset:
    """Load Mozilla Common Voice (requires HF login for some versions)."""
    ds = load_dataset(
        "mozilla-foundation/common_voice_16_1",
        language,
        split=split,
        trust_remote_code=True,
    )
    ds = ds.cast_column("audio", Audio(sampling_rate=TARGET_SR))
    ds = ds.map(lambda b: {"dataset": "common_voice", "text": b["sentence"].lower()})
    return ds


def merge_datasets(dataset_list: list[Dataset]) -> Dataset:
    """Concatenate multiple datasets into one, keeping only common columns."""
    common_cols = {"audio", "text", "speaker_id", "dataset"}
    cleaned = []
    for ds in dataset_list:
        keep = [c for c in ds.column_names if c in common_cols]
        ds = ds.select_columns(keep)
        if "speaker_id" not in ds.column_names:
            ds = ds.map(lambda b: {"speaker_id": "unknown"})
        cleaned.append(ds)
    return concatenate_datasets(cleaned)
