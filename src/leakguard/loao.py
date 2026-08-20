"""Leave-One-Attack-Out (LOAO) zero-day generalisation protocol (Section 5.5).

For each held-out attack category, the probe is trained on benign traffic plus all
*other* attack categories, then its recall on the held-out (unseen) category is
measured. This reproduces Table 9:

    UNSW-NB15  DoS    -> 98.67% recall
    UNSW-NB15  Worms  -> 94.44% recall (18 test samples)
    TON_IoT    mitm   -> 38.00% recall (200 held-out records)
    TON_IoT    ddos   -> 84.57% recall

Recommended in the manuscript as an additional reporting standard for NIDS studies
claiming robustness to novel attacks.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .preprocessing import LeakageAwarePreprocessor, PreprocessConfig
from .train import TrainConfig, train_probe


def _category_series(frame: pd.DataFrame) -> pd.Series | None:
    for col in ("type", "attack_cat", "attack_category"):
        if col in frame.columns:
            return frame[col].astype(str).str.lower()
    return None


def run_loao(
    frame: pd.DataFrame,
    label_col: str,
    categories: list[str],
    train_config: TrainConfig | None = None,
    preprocess_config: PreprocessConfig | None = None,
) -> list[dict[str, float]]:
    """Run LOAO for each requested held-out attack category.

    Returns one record per category with the recall on unseen attacks.
    """
    cat = _category_series(frame)
    if cat is None:
        raise ValueError("no attack-category column found for LOAO")

    results: list[dict[str, float]] = []
    for held in categories:
        held = held.lower()
        is_held = cat == held
        train_mask = ~is_held  # benign + all other attack categories
        test_mask = is_held    # unseen attacks only; recall is the target

        if test_mask.sum() == 0:
            continue

        train_df = frame[train_mask].copy()
        test_df = frame[test_mask].copy()

        pre = LeakageAwarePreprocessor(preprocess_config or PreprocessConfig())
        X_tr, y_tr = pre.fit_transform(train_df, label_col)
        X_te, y_te = pre.transform(test_df, label_col)

        # held-out rows are all attacks; recall = fraction predicted as attack
        metrics = train_probe(X_tr, y_tr, X_te, np.ones_like(y_te), train_config)
        results.append(
            {
                "category": held,
                "test_samples": int(test_mask.sum()),
                "unseen_recall": float(metrics["attack_recall"]),
            }
        )
    return results
