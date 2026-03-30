#!/usr/bin/env python3
"""
Fine-tune the CTC head of wav2vec2 on pronunciation assessment data.

Freezes the wav2vec2 encoder and only trains the linear CTC projection head.
Uses our diverse audio dataset with Azure transcriptions as labels.

This demonstrates the training pipeline even on CPU with limited data.
"""

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
CHECKPOINT_DIR = Path("checkpoints")


def load_training_data(processor):
    diverse_dir = Path("data/diverse_audio")
    benchmark_dir = Path("data/benchmark_audio")

    csv_path = RESULTS_DIR / "benchmark_diverse.csv"
    df = pd.read_csv(csv_path) if csv_path.exists() else pd.DataFrame()

    csv_path2 = RESULTS_DIR / "benchmark.csv"
    df2 = pd.read_csv(csv_path2) if csv_path2.exists() else pd.DataFrame()

    samples = []

    for _, row in df.iterrows():
        if pd.isna(row.get("azure_rec")):
            continue
        text = row["text"]
        quality = row.get("quality", "native")
        safe_name = text.lower().replace(" ", "_")[:40]
        audio_path = diverse_dir / f"{safe_name}_{quality}.wav"
        if audio_path.exists():
            samples.append({
                "audio_path": str(audio_path),
                "text": str(row["azure_rec"]).strip().rstrip(".").upper(),
                "quality": quality,
            })

    for _, row in df2.iterrows():
        if pd.isna(row.get("azure_recognised")):
            continue
        text = row["text"]
        safe_name = text.lower().replace(" ", "_")[:40]
        audio_path = benchmark_dir / f"{safe_name}.wav"
        if audio_path.exists():
            samples.append({
                "audio_path": str(audio_path),
                "text": str(row["azure_recognised"]).strip().rstrip(".").upper(),
                "quality": "native",
            })

    logger.info("Loaded %d training samples", len(samples))
    return samples


def prepare_batch(samples, processor, max_samples=None):
    if max_samples:
        samples = samples[:max_samples]

    input_values_list = []
    labels_list = []

    for s in samples:
        text = s["text"].strip()
        if not text:
            continue

        audio, sr = sf.read(s["audio_path"], dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)

        inputs = processor(audio, sampling_rate=sr, return_tensors="pt", padding=False)

        label_ids = processor.tokenizer(text).input_ids
        if not label_ids:
            continue

        input_values_list.append(inputs.input_values.squeeze(0))
        labels_list.append(torch.tensor(label_ids))

    return input_values_list, labels_list


def train_epoch(model, input_values_list, labels_list, optimizer, epoch):
    model.train()
    total_loss = 0
    n_samples = len(input_values_list)

    indices = list(range(n_samples))
    np.random.shuffle(indices)

    for i, idx in enumerate(indices):
        optimizer.zero_grad()

        input_values = input_values_list[idx].unsqueeze(0)
        labels = labels_list[idx].unsqueeze(0)

        labels_padded = labels.clone()
        labels_padded[labels_padded == model.config.pad_token_id] = -100

        outputs = model(input_values=input_values, labels=labels_padded)
        loss = outputs.loss

        if loss is None or torch.isnan(loss):
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        if (i + 1) % 10 == 0:
            logger.info("  Epoch %d [%d/%d] loss=%.4f", epoch, i + 1, n_samples, loss.item())

    avg_loss = total_loss / max(n_samples, 1)
    return avg_loss


def evaluate(model, processor, input_values_list, labels_list):
    model.eval()
    total_loss = 0
    correct_chars = 0
    total_chars = 0

    with torch.no_grad():
        for i in range(len(input_values_list)):
            input_values = input_values_list[i].unsqueeze(0)
            labels = labels_list[i].unsqueeze(0)
            labels_padded = labels.clone()
            labels_padded[labels_padded == model.config.pad_token_id] = -100

            outputs = model(input_values=input_values, labels=labels_padded)
            if outputs.loss is not None:
                total_loss += outputs.loss.item()

            pred_ids = outputs.logits.argmax(dim=-1)
            pred_text = processor.batch_decode(pred_ids)[0]
            ref_text = processor.tokenizer.decode(labels[0].tolist(), group_tokens=False)

            for p, r in zip(pred_text[:len(ref_text)], ref_text):
                if p == r:
                    correct_chars += 1
                total_chars += 1

    avg_loss = total_loss / max(len(input_values_list), 1)
    char_acc = correct_chars / max(total_chars, 1)
    return avg_loss, char_acc


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    model_name = os.getenv("SPEAKRIGHT_MODEL", "facebook/wav2vec2-large-960h")
    logger.info("Loading %s ...", model_name)

    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)

    n_frozen = 0
    for name, param in model.named_parameters():
        if "lm_head" not in name:
            param.requires_grad = False
            n_frozen += 1
        else:
            param.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info("Frozen: %d layers, Trainable: %d params (%.1f%% of %dM)",
                n_frozen, trainable, trainable / total * 100, total // 1_000_000)

    samples = load_training_data(processor)
    if not samples:
        logger.error("No training data found")
        return

    np.random.seed(42)
    np.random.shuffle(samples)
    split = int(len(samples) * 0.8)
    train_samples = samples[:split]
    val_samples = samples[split:]
    logger.info("Train: %d, Val: %d", len(train_samples), len(val_samples))

    logger.info("Preparing training batches...")
    train_inputs, train_labels = prepare_batch(train_samples, processor)
    val_inputs, val_labels = prepare_batch(val_samples, processor)

    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4,
        weight_decay=0.01,
    )

    n_epochs = 5
    history = {"train_loss": [], "val_loss": [], "val_char_acc": []}

    logger.info("Training CTC head for %d epochs...", n_epochs)
    for epoch in range(1, n_epochs + 1):
        train_loss = train_epoch(model, train_inputs, train_labels, optimizer, epoch)
        val_loss, val_acc = evaluate(model, processor, val_inputs, val_labels)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_char_acc"].append(val_acc)

        logger.info("Epoch %d: train_loss=%.4f  val_loss=%.4f  val_char_acc=%.2f%%",
                     epoch, train_loss, val_loss, val_acc * 100)

    checkpoint_path = CHECKPOINT_DIR / "wav2vec2_finetuned_ctc"
    model.save_pretrained(str(checkpoint_path))
    processor.save_pretrained(str(checkpoint_path))
    logger.info("Checkpoint saved to %s", checkpoint_path)

    pre_loss, pre_acc = evaluate(
        Wav2Vec2ForCTC.from_pretrained(model_name), processor, val_inputs, val_labels
    )

    print("\n" + "=" * 60)
    print("FINE-TUNING RESULTS (CTC Head Only)")
    print("=" * 60)
    print(f"  Model:           {model_name}")
    print(f"  Trainable params: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")
    print(f"  Train samples:   {len(train_samples)}")
    print(f"  Val samples:     {len(val_samples)}")
    print(f"  Epochs:          {n_epochs}")
    print(f"  Pre-FT val loss:  {pre_loss:.4f}  char_acc={pre_acc*100:.1f}%")
    print(f"  Post-FT val loss: {history['val_loss'][-1]:.4f}  char_acc={history['val_char_acc'][-1]*100:.1f}%")
    print(f"  Improvement:     {(history['val_char_acc'][-1] - pre_acc)*100:+.1f}% char accuracy")
    print("=" * 60)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(range(1, n_epochs + 1), history["train_loss"], "b-o", label="Train Loss")
    axes[0].plot(range(1, n_epochs + 1), history["val_loss"], "r-o", label="Val Loss")
    axes[0].axhline(y=pre_loss, color="gray", linestyle="--", label=f"Pre-FT Val Loss ({pre_loss:.3f})")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("CTC Loss")
    axes[0].set_title("Training Loss", fontweight="bold")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(range(1, n_epochs + 1), [a * 100 for a in history["val_char_acc"]], "g-o", label="Val Char Accuracy")
    axes[1].axhline(y=pre_acc * 100, color="gray", linestyle="--", label=f"Pre-FT ({pre_acc*100:.1f}%)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Character Accuracy (%)")
    axes[1].set_title("Validation Accuracy", fontweight="bold")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.suptitle("CTC Head Fine-Tuning on Pronunciation Data", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(RESULTS_DIR / "finetuning_curves.png"), dpi=150, bbox_inches="tight")

    with open(RESULTS_DIR / "finetuning_report.json", "w") as f:
        json.dump({
            "model": model_name,
            "trainable_params": trainable,
            "total_params": total,
            "n_train": len(train_samples),
            "n_val": len(val_samples),
            "epochs": n_epochs,
            "pre_ft_val_loss": pre_loss,
            "pre_ft_char_acc": pre_acc,
            "post_ft_val_loss": history["val_loss"][-1],
            "post_ft_char_acc": history["val_char_acc"][-1],
            "history": history,
        }, f, indent=2)

    logger.info("Fine-tuning complete!")


if __name__ == "__main__":
    main()
