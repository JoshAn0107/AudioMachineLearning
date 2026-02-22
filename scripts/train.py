"""
Fine-tune wav2vec2 for pronunciation-aware phoneme recognition.

This script trains a CTC head on top of a frozen wav2vec2 feature extractor
using TIMIT (phoneme labels) or force-aligned LibriSpeech/VCTK.

Usage:
    python scripts/train.py \
        --dataset timit \
        --model facebook/wav2vec2-base \
        --output-dir checkpoints/wav2vec2-pron-v1 \
        --epochs 20 \
        --batch-size 16

Then set SPEAKRIGHT_MODEL=checkpoints/wav2vec2-pron-v1 when running the API.
"""

import argparse
import logging
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import (
    Trainer,
    TrainingArguments,
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def compute_ctc_loss(pred):
    """HuggingFace Trainer compute_metrics hook for CTC models."""
    from datasets import load_metric
    wer_metric = load_metric("wer")
    logits = pred.predictions
    labels = pred.label_ids

    import numpy as np
    pred_ids = np.argmax(logits, axis=-1)
    # Replace -100 (padding) with pad token id before decoding
    labels[labels == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(labels, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


def prepare_dataset(batch):
    """Tokenise audio + labels for the Trainer."""
    audio = batch["audio"]
    inputs = processor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
        return_tensors="pt",
        padding=True,
    )
    batch["input_values"] = inputs.input_values[0]
    batch["input_length"] = len(inputs.input_values[0])

    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
    return batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="timit", help="Dataset name under data/processed/")
    parser.add_argument("--split", default="train", help="Training split name")
    parser.add_argument("--model", default="facebook/wav2vec2-base")
    parser.add_argument("--output-dir", default="checkpoints/wav2vec2-pron-v1")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--freeze-feature-extractor", action="store_true", default=True)
    args = parser.parse_args()

    global processor
    logger.info("Loading processor and model: %s", args.model)
    processor = Wav2Vec2Processor.from_pretrained(args.model)
    model = Wav2Vec2ForCTC.from_pretrained(
        args.model,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
    )

    if args.freeze_feature_extractor:
        model.freeze_feature_extractor()
        logger.info("Feature extractor frozen. Only transformer layers will be trained.")

    ds_path = Path("data/processed") / args.dataset / args.split
    logger.info("Loading dataset from %s", ds_path)
    train_ds = load_from_disk(str(ds_path))
    train_ds = train_ds.map(prepare_dataset, remove_columns=train_ds.column_names, num_proc=4)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=500,
        save_steps=500,
        eval_strategy="no",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        group_by_length=True,          # speeds up training by batching similar lengths
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
    )

    logger.info("Starting training …")
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    logger.info("Model saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
