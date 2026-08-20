"""Evaluation metrics reported in the manuscript (Tables 4, 12).

All metrics are computed on the binary attack/benign task. ``attack_*`` metrics are
reported for the positive (attack) class specifically, and FPR is reported because
Section 5 argues headline accuracy alone is not decision-relevant.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return the metric bundle used throughout the results tables."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    tn, fp, fn, tp = _safe_confusion(y_true, y_pred)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "accuracy": float(acc),
        "attack_precision": float(prec),
        "attack_recall": float(rec),
        "attack_f1": float(f1),
        "fpr": float(fpr),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def _safe_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return int(tn), int(fp), int(fn), int(tp)


def aggregate_seeds(runs: list[dict[str, float]]) -> dict[str, float]:
    """Mean +/- std across seed repetitions for each metric key."""
    if not runs:
        return {}
    keys = [k for k in runs[0] if isinstance(runs[0][k], (int, float))]
    out: dict[str, float] = {}
    for k in keys:
        vals = np.array([r[k] for r in runs], dtype=float)
        out[f"{k}_mean"] = float(vals.mean())
        out[f"{k}_std"] = float(vals.std(ddof=0))
    return out
