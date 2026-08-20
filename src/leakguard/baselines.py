"""Same-pipeline baselines (Section 5.1, Table 4).

These run under the *identical* leakage-aware pipeline as the DCFA-Net probe so the
comparison is fair. The manuscript reports that a same-pipeline random forest exceeds
DCFA-Net on all three benchmarks -- a required property of a measurement instrument,
not a defect.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .metrics import compute_metrics


def run_baseline(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 42,
) -> dict[str, float]:
    """Fit a baseline classifier and return its metric bundle."""
    name = name.lower()
    if name in ("rf", "random_forest", "randomforest"):
        clf = RandomForestClassifier(
            n_estimators=200,
            n_jobs=-1,
            random_state=random_state,
        )
    elif name in ("lr", "logreg", "logistic_regression"):
        clf = LogisticRegression(max_iter=1000, random_state=random_state)
    else:
        raise ValueError(f"unknown baseline {name!r}; expected 'rf' or 'lr'")

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return compute_metrics(y_test, y_pred)
