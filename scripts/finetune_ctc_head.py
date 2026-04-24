"""
Wav2Vec2 fine-tuning with last-2-encoder-layers + CTC head unfrozen.

Fixes three issues found in the previous run:
  1. NaN CTC loss on every sample (head-only training was unstable).
  2. Zero training loss (gradients never flowed).
  3. Overestimated char accuracy (old metric used length-truncated zip).

This version:
  - Unfreezes the final 2 encoder layers + the CTC head only.
  - Disables dropout and spec-augment during training (small-data regime).
  - Uses an edit-distance character error rate (CER) as the validation metric.
  - Enforces a sanity check that both training loss and val metrics changed.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import torch.nn.functional as F
from torch.optim import AdamW
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("finetune")

RESULTS_DIR = Path("results")
CHECKPOINT_DIR = Path("checkpoints") / "wav2vec2_finetuned_ctc_v2"
MIN_TRAIN_LOSS_SUM = 0.1
LR = 3e-5
EPOCHS = 4
HOP = 320
MIN_FRAMES_PER_LABEL = 3

MODEL_NAME = os.environ.get("SPEAKRIGHT_MODEL", "facebook/wav2vec2-large-960h")


def edit_distance(a: str, b: str) -> int:
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        curr = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + cost,
            )
        prev = curr
    return prev[lb]


def char_accuracy(pred: str, ref: str) -> float:
    if len(ref) == 0:
        return 1.0 if len(pred) == 0 else 0.0
    dist = edit_distance(pred.strip().upper(), ref.strip().upper())
    return max(0.0, 1.0 - dist / max(len(ref), 1))


def load_audio(path: str):
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


def build_dataset(processor):
    diverse = Path("data/diverse_audio")
    benchmark = Path("data/benchmark_audio")

    rows = []
    for csv_path, audio_root, ref_col in [
        (RESULTS_DIR / "benchmark_diverse.csv", diverse, "azure_rec"),
        (RESULTS_DIR / "benchmark.csv", benchmark, "azure_recognised"),
    ]:
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            text_col = row.get(ref_col)
            if pd.isna(text_col) or not str(text_col).strip():
                continue
            ref = str(text_col).strip().rstrip(".").upper()
            if not ref:
                continue

            text = row["text"]
            safe_name = str(text).lower().replace(" ", "_")[:40]
            if "quality" in df.columns:
                quality = row.get("quality", "native")
                audio_path = audio_root / f"{safe_name}_{quality}.wav"
            else:
                audio_path = audio_root / f"{safe_name}.wav"

            if not audio_path.exists():
                continue

            try:
                audio, sr = load_audio(str(audio_path))
            except Exception:
                continue

            label_ids = processor.tokenizer(ref, return_tensors=None).input_ids
            if not label_ids:
                continue

            n_frames = max(1, len(audio) // HOP)
            if n_frames < MIN_FRAMES_PER_LABEL * len(label_ids):
                continue

            rows.append({
                "audio_path": str(audio_path),
                "text": ref,
                "n_samples": len(audio),
                "label_len": len(label_ids),
            })

    log.info("Assembled %d usable samples after filtering.", len(rows))
    return rows


def configure_trainable(model: Wav2Vec2ForCTC):
    total = 0
    trainable = 0
    for name, p in model.named_parameters():
        total += p.numel()
        is_trainable = False
        if name.startswith("lm_head"):
            is_trainable = True
        if ".layers." in name:
            try:
                idx = int(name.split(".layers.")[1].split(".")[0])
            except ValueError:
                idx = -1
            n_layers = len(model.wav2vec2.encoder.layers)
            if idx >= n_layers - 2:
                is_trainable = True
        p.requires_grad = is_trainable
        if is_trainable:
            trainable += p.numel()

    if hasattr(model.config, "apply_spec_augment"):
        model.config.apply_spec_augment = False
    if hasattr(model.config, "hidden_dropout"):
        model.config.hidden_dropout = 0.0
    if hasattr(model.config, "attention_dropout"):
        model.config.attention_dropout = 0.0
    if hasattr(model.config, "feat_proj_dropout"):
        model.config.feat_proj_dropout = 0.0
    if hasattr(model.config, "final_dropout"):
        model.config.final_dropout = 0.0
    if hasattr(model.config, "ctc_loss_reduction"):
        model.config.ctc_loss_reduction = "mean"

    for mod in model.modules():
        if isinstance(mod, torch.nn.Dropout):
            mod.p = 0.0

    return total, trainable


def forward_loss(model, processor, sample, device):
    audio, _ = load_audio(sample["audio_path"])
    inputs = processor(
        audio,
        sampling_rate=16000,
        return_tensors="pt",
        padding=False,
    )
    input_values = inputs.input_values.to(device)

    labels = torch.tensor(
        processor.tokenizer(sample["text"]).input_ids,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)

    outputs = model(input_values=input_values, labels=labels)
    return outputs.loss, outputs.logits


def decode_greedy(processor, logits):
    pred_ids = logits.argmax(dim=-1)
    return processor.batch_decode(pred_ids)[0].strip().upper()


def evaluate(model, processor, samples, device, label=""):
    model.eval()
    losses = []
    accs = []
    with torch.no_grad():
        for s in samples:
            try:
                loss, logits = forward_loss(model, processor, s, device)
            except Exception as exc:
                log.warning("%s: skipping %s (%s)", label, s["audio_path"], exc)
                continue
            if loss is None or torch.isnan(loss) or torch.isinf(loss):
                continue
            pred = decode_greedy(processor, logits)
            accs.append(char_accuracy(pred, s["text"]))
            losses.append(loss.item())
    if not losses:
        return float("nan"), float("nan")
    return float(np.mean(losses)), float(np.mean(accs))


def train_epoch(model, processor, samples, optimizer, device, epoch):
    model.train()
    losses = []
    skipped = 0
    idx_order = list(range(len(samples)))
    np.random.shuffle(idx_order)
    for i, idx in enumerate(idx_order):
        s = samples[idx]
        optimizer.zero_grad()
        try:
            loss, _ = forward_loss(model, processor, s, device)
        except Exception as exc:
            log.warning("epoch %d: forward error on %s (%s)", epoch, s["audio_path"], exc)
            skipped += 1
            continue
        if loss is None or torch.isnan(loss) or torch.isinf(loss):
            skipped += 1
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )
        optimizer.step()
        losses.append(loss.item())
        if (i + 1) % 10 == 0:
            log.info(
                "epoch %d [%d/%d] loss=%.4f skipped=%d",
                epoch, i + 1, len(samples), loss.item(), skipped,
            )
    mean_loss = float(np.mean(losses)) if losses else 0.0
    log.info(
        "epoch %d done: mean_loss=%.4f, usable=%d, skipped=%d",
        epoch, mean_loss, len(losses), skipped,
    )
    return mean_loss, skipped


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    log.info("Loading %s on %s", MODEL_NAME, device)
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME).to(device)

    total, trainable = configure_trainable(model)
    log.info(
        "Total params=%d, trainable=%d (%.2f%%)",
        total, trainable, 100 * trainable / max(total, 1),
    )

    samples = build_dataset(processor)
    if len(samples) < 20:
        raise RuntimeError(f"Not enough usable samples ({len(samples)})")

    rng = np.random.default_rng(42)
    idx = np.arange(len(samples))
    rng.shuffle(idx)
    cut = int(len(samples) * 0.8)
    train_samples = [samples[i] for i in idx[:cut]]
    val_samples = [samples[i] for i in idx[cut:]]
    log.info("train=%d val=%d", len(train_samples), len(val_samples))

    pre_val_loss, pre_val_acc = evaluate(model, processor, val_samples, device, label="pre")
    log.info("PRE: val_loss=%.4f val_char_acc=%.4f", pre_val_loss, pre_val_acc)

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR,
        weight_decay=0.0,
    )

    history = {"train_loss": [], "val_loss": [], "val_char_acc": [], "skipped": []}
    partial_path = RESULTS_DIR / "finetuning_report.json"
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, skipped = train_epoch(model, processor, train_samples, optimizer, device, epoch)
        val_loss, val_acc = evaluate(model, processor, val_samples, device, label=f"epoch{epoch}")
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_char_acc"].append(val_acc)
        history["skipped"].append(skipped)
        log.info(
            "epoch %d summary: train_loss=%.4f val_loss=%.4f val_char_acc=%.4f skipped=%d duration=%.1fs",
            epoch, train_loss, val_loss, val_acc, skipped, time.time() - t0,
        )

        partial_report = {
            "model": MODEL_NAME,
            "total_params": total,
            "trainable_params": trainable,
            "n_train": len(train_samples),
            "n_val": len(val_samples),
            "epochs_completed": epoch,
            "epochs_planned": EPOCHS,
            "learning_rate": LR,
            "strategy": "last-2 encoder layers + lm_head unfrozen",
            "pre_ft_val_loss": pre_val_loss,
            "pre_ft_char_acc": pre_val_acc,
            "post_ft_val_loss": val_loss,
            "post_ft_char_acc": val_acc,
            "train_loss_sum_so_far": float(sum(l for l in history["train_loss"] if not np.isnan(l))),
            "history": history,
            "status": "in_progress",
        }
        with partial_path.open("w") as fh:
            json.dump(partial_report, fh, indent=2)

    post_val_loss, post_val_acc = evaluate(model, processor, val_samples, device, label="post")
    log.info("POST: val_loss=%.4f val_char_acc=%.4f", post_val_loss, post_val_acc)

    train_loss_sum = float(sum(l for l in history["train_loss"] if not np.isnan(l)))
    loss_moved = (
        not np.isnan(pre_val_loss)
        and not np.isnan(post_val_loss)
        and abs(post_val_loss - pre_val_loss) > 1e-4
    )
    acc_moved = (
        not np.isnan(pre_val_acc)
        and not np.isnan(post_val_acc)
        and abs(post_val_acc - pre_val_acc) > 1e-4
    )
    sanity_passed = train_loss_sum > MIN_TRAIN_LOSS_SUM and (loss_moved or acc_moved)

    report = {
        "model": MODEL_NAME,
        "total_params": total,
        "trainable_params": trainable,
        "n_train": len(train_samples),
        "n_val": len(val_samples),
        "epochs": EPOCHS,
        "learning_rate": LR,
        "strategy": "last-2 encoder layers + lm_head unfrozen",
        "pre_ft_val_loss": pre_val_loss,
        "pre_ft_char_acc": pre_val_acc,
        "post_ft_val_loss": post_val_loss,
        "post_ft_char_acc": post_val_acc,
        "train_loss_sum": train_loss_sum,
        "sanity_check_passed": bool(sanity_passed),
        "history": history,
    }

    with (RESULTS_DIR / "finetuning_report.json").open("w") as fh:
        json.dump(report, fh, indent=2)
    log.info("Wrote finetuning_report.json")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    xs = list(range(1, EPOCHS + 1))
    axes[0].plot(xs, history["train_loss"], marker="o", label="train")
    axes[0].plot(xs, history["val_loss"], marker="s", label="val")
    axes[0].axhline(pre_val_loss, linestyle="--", color="gray", label="pre-FT val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("CTC loss")
    axes[0].set_title("Fine-tune loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(xs, history["val_char_acc"], marker="o", color="green", label="val char acc (1-CER)")
    axes[1].axhline(pre_val_acc, linestyle="--", color="gray", label="pre-FT val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Char Accuracy")
    axes[1].set_title("Fine-tune val char accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "finetuning_curves.png", dpi=150, bbox_inches="tight")
    log.info("Wrote finetuning_curves.png")

    if not sanity_passed:
        log.error("Sanity check FAILED: train_loss_sum=%.4f loss_moved=%s acc_moved=%s",
                  train_loss_sum, loss_moved, acc_moved)
        sys.exit(2)

    log.info("Sanity check PASSED.")


if __name__ == "__main__":
    main()
