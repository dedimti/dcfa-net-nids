"""One-command LeakGuard audit runner.

Runs the full single-architecture attribution: fixes the DCFA-Net probe and varies
one evaluation factor at a time (deduplication, label-determining-field guard,
identifier encoding, checkpoint policy, architectural variant), reporting each
factor's marginal effect on a common scale.

Usage:
    python -m leakguard.audit --dataset ton-iot --seeds 3 --synthetic
    leakguard-audit --dataset unsw-nb15 --out results.json
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace

import numpy as np

from .baselines import run_baseline
from .data import load_dataset
from .metrics import aggregate_seeds
from .preprocessing import LeakageAwarePreprocessor, PreprocessConfig
from .train import TrainConfig, train_probe


def _split(X: np.ndarray, y: np.ndarray, seed: int, test_frac: float = 0.3):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))  # shuffle: datasets ship class-ordered
    n_test = int(len(X) * test_frac)
    te, tr = idx[:n_test], idx[n_test:]
    return X[tr], y[tr], X[te], y[te]


def _prep(frame, label_col, cfg: PreprocessConfig, seed: int):
    frame = frame.copy()
    pre = LeakageAwarePreprocessor(cfg)
    frame = pre.deduplicate(frame)
    X, y = pre.fit_transform(frame, label_col)
    return _split(X, y, seed)


def run_audit(dataset: str, seeds: int = 3, synthetic: bool = False) -> dict:
    bundle = load_dataset(dataset, use_synthetic=synthetic)
    frame, label_col = bundle.frame, bundle.label_col

    base_pre = PreprocessConfig()
    base_train = TrainConfig(epochs=8 if synthetic else 30)

    report: dict[str, object] = {"dataset": dataset, "factors": {}}

    # --- probe reference (leakage-aware, best checkpoint, full variant) ------
    report["factors"]["probe_reference"] = _sweep_seeds(
        frame, label_col, base_pre, base_train, seeds
    )

    # --- factor: deduplication ----------------------------------------------
    report["factors"]["no_dedup"] = _sweep_seeds(
        frame, label_col, replace(base_pre, deduplicate=False), base_train, seeds
    )

    # --- factor: identifier encoding (IP leakage) ---------------------------
    report["factors"]["encode_identifiers"] = _sweep_seeds(
        frame, label_col, replace(base_pre, encode_identifiers=True), base_train, seeds
    )

    # --- factor: label-determining field kept -------------------------------
    report["factors"]["keep_label_field"] = _sweep_seeds(
        frame, label_col, replace(base_pre, drop_label_determining=False), base_train, seeds
    )

    # --- factor: checkpoint policy ------------------------------------------
    report["factors"]["checkpoint_last"] = _sweep_seeds(
        frame, label_col, base_pre, replace(base_train, checkpoint="last"), seeds
    )

    # --- factor: architectural variant (Section 5.4 ablation) ---------------
    for variant in ("local", "global"):
        report["factors"][f"variant_{variant}"] = _sweep_seeds(
            frame, label_col, base_pre, replace(base_train, variant=variant), seeds
        )

    # --- same-pipeline baselines --------------------------------------------
    Xtr, ytr, Xte, yte = _prep(frame, label_col, base_pre, seed=0)
    report["baselines"] = {
        "random_forest": run_baseline("rf", Xtr, ytr, Xte, yte),
        "logistic_regression": run_baseline("lr", Xtr, ytr, Xte, yte),
    }
    return report


def _sweep_seeds(frame, label_col, pre_cfg, train_cfg, seeds: int) -> dict:
    runs = []
    for s in range(seeds):
        Xtr, ytr, Xte, yte = _prep(frame, label_col, pre_cfg, seed=s)
        runs.append(train_probe(Xtr, ytr, Xte, yte, replace(train_cfg, seed=s)))
    return aggregate_seeds(runs)


def _to_native(obj):
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def main() -> None:
    ap = argparse.ArgumentParser(description="LeakGuard audit runner")
    ap.add_argument("--dataset", required=True, choices=["unsw-nb15", "cic-ids2017", "ton-iot"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--synthetic", action="store_true", help="use offline synthetic data")
    ap.add_argument("--out", default=None, help="write JSON report to this path")
    args = ap.parse_args()

    report = run_audit(args.dataset, seeds=args.seeds, synthetic=args.synthetic)
    report = _to_native(report)
    text = json.dumps(report, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
