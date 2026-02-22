"""
Evaluation metrics for comparing SpeakRight vs Azure and against ground truth.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def pearson_r(y_true: list[float], y_pred: list[float]) -> float:
    """Pearson correlation coefficient between two score lists."""
    a, b = np.array(y_true, dtype=float), np.array(y_pred, dtype=float)
    if a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def score_mae(y_true: list[float], y_pred: list[float]) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def score_rmse(y_true: list[float], y_pred: list[float]) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def error_detection_metrics(
    true_errors: list[str],
    pred_errors: list[str],
    positive_label: str = "Mispronunciation",
) -> dict:
    """
    Precision / recall / F1 for mispronunciation detection at word level.

    Args:
        true_errors: list of ground-truth ErrorType strings per word.
        pred_errors: list of predicted ErrorType strings per word.
        positive_label: the class considered as "error".
    """
    tp = fp = fn = 0
    for t, p in zip(true_errors, pred_errors):
        t_pos = t == positive_label
        p_pos = p == positive_label
        if t_pos and p_pos:
            tp += 1
        elif not t_pos and p_pos:
            fp += 1
        elif t_pos and not p_pos:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def summarise_comparison(records: list[dict]) -> dict:
    """
    Given a list of comparison records (one per audio sample), compute
    aggregate metrics between SpeakRight and Azure.

    Each record must have:
        - speakright_pron  (float)
        - azure_pron       (float)
        - speakright_acc   (float)
        - azure_acc        (float)
    """
    sr_pron = [r["speakright_pron"] for r in records]
    az_pron = [r["azure_pron"] for r in records]
    sr_acc = [r["speakright_acc"] for r in records]
    az_acc = [r["azure_acc"] for r in records]

    return {
        "n_samples": len(records),
        "pron_score": {
            "pearson_r": pearson_r(az_pron, sr_pron),
            "mae": score_mae(az_pron, sr_pron),
            "rmse": score_rmse(az_pron, sr_pron),
            "speakright_mean": float(np.mean(sr_pron)),
            "azure_mean": float(np.mean(az_pron)),
        },
        "accuracy_score": {
            "pearson_r": pearson_r(az_acc, sr_acc),
            "mae": score_mae(az_acc, sr_acc),
            "rmse": score_rmse(az_acc, sr_acc),
        },
    }
