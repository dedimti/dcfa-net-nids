"""Unit tests for the LeakGuard reference implementation.

Run: pytest -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from leakguard import (
    DCFANet,
    LeakageAwarePreprocessor,
    PreprocessConfig,
    aggregate_seeds,
    build_model,
    compute_metrics,
    load_dataset,
    measure_train_test_overlap,
    run_baseline,
    train_probe,
)
from leakguard.train import TrainConfig

# -- model -----------------------------------------------------------------

def test_model_forward_shape():
    model = build_model(n_features=12, n_classes=2)
    x = torch.randn(16, 12)
    out = model(x)
    assert out.shape == (16, 2)


def test_model_variants_run():
    x = torch.randn(8, 10)
    for variant in DCFANet.VARIANTS:
        model = DCFANet(n_features=10, variant=variant)
        assert model(x).shape == (8, 2)


def test_model_rejects_bad_variant():
    try:
        DCFANet(n_features=5, variant="nope")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown variant")


# -- preprocessing ----------------------------------------------------------

def _toy_frame(n=200, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    return pd.DataFrame(
        {
            "f0": rng.normal(size=n),
            "f1": rng.normal(size=n),
            "proto": rng.choice(["tcp", "udp"], size=n),
            "src_ip": rng.choice(["1.1.1.1", "2.2.2.2"], size=n),
            "type": np.where(y == 1, "ddos", "normal"),
            "label": y,
        }
    )


def test_dedup_removes_duplicates():
    df = _toy_frame(100)
    df = pd.concat([df, df.iloc[:20]], ignore_index=True)
    pre = LeakageAwarePreprocessor(PreprocessConfig(deduplicate=True))
    out = pre.deduplicate(df)
    assert len(out) < len(df)
    assert pre.n_duplicates_removed_ >= 20


def test_identifier_dropped_by_default():
    df = _toy_frame()
    pre = LeakageAwarePreprocessor(PreprocessConfig(encode_identifiers=False))
    X, y = pre.fit_transform(df, "label")
    assert "src_ip" not in pre.feature_names_


def test_identifier_kept_when_encoded():
    df = _toy_frame()
    pre = LeakageAwarePreprocessor(PreprocessConfig(encode_identifiers=True))
    X, y = pre.fit_transform(df, "label")
    assert "src_ip" in pre.feature_names_


def test_label_determining_field_dropped():
    df = _toy_frame()
    pre = LeakageAwarePreprocessor(PreprocessConfig(drop_label_determining=True))
    X, y = pre.fit_transform(df, "label")
    assert "type" not in pre.feature_names_


def test_binary_label_encoding():
    pre = LeakageAwarePreprocessor()
    y = pre._encode_label(np.array(["normal", "ddos", "benign", "mitm"]))
    assert list(y) == [0, 1, 0, 1]


# -- metrics ----------------------------------------------------------------

def test_compute_metrics_perfect():
    y = np.array([0, 1, 0, 1])
    m = compute_metrics(y, y)
    assert m["accuracy"] == 1.0
    assert m["fpr"] == 0.0


def test_fpr_computation():
    y_true = np.array([0, 0, 0, 1])
    y_pred = np.array([1, 0, 0, 1])  # one false positive of three benign
    m = compute_metrics(y_true, y_pred)
    assert abs(m["fpr"] - (1 / 3)) < 1e-9


def test_aggregate_seeds():
    runs = [{"accuracy": 0.9}, {"accuracy": 1.0}]
    agg = aggregate_seeds(runs)
    assert abs(agg["accuracy_mean"] - 0.95) < 1e-9
    assert agg["accuracy_std"] > 0


# -- overlap ----------------------------------------------------------------

def test_train_test_overlap():
    train = _toy_frame(100, seed=1)
    test = train.iloc[:30].copy()
    res = measure_train_test_overlap(train, test)
    assert res["overlap_fraction"] > 0.9


# -- integration ------------------------------------------------------------

def test_synthetic_load_and_train():
    bundle = load_dataset("ton-iot", use_synthetic=True)
    pre = LeakageAwarePreprocessor()
    frame = pre.deduplicate(bundle.frame)
    X, y = pre.fit_transform(frame, bundle.label_col)
    n = len(X)
    cut = int(n * 0.7)
    m = train_probe(
        X[:cut], y[:cut], X[cut:], y[cut:], TrainConfig(epochs=3, seed=0)
    )
    assert 0.0 <= m["accuracy"] <= 1.0


def test_baseline_runs():
    bundle = load_dataset("unsw-nb15", use_synthetic=True)
    pre = LeakageAwarePreprocessor()
    X, y = pre.fit_transform(bundle.frame, bundle.label_col)
    cut = int(len(X) * 0.7)
    m = run_baseline("rf", X[:cut], y[:cut], X[cut:], y[cut:])
    assert 0.0 <= m["accuracy"] <= 1.0
