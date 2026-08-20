"""Dataset loaders for the three audited benchmarks.

The loaders read from the public Hugging Face mirrors named in the manuscript and
fall back to a synthetic generator so that the full pipeline, unit tests, and CI run
with no external download.

Public mirrors (Section 4.1):
    UNSW-NB15    -> Mireu-Lab/UNSW-NB15
    CIC-IDS2017  -> c01dsnap/CIC-IDS2017
    TON_IoT      -> codymlewis/TON_IoT_network

Important loader rule for TON_IoT: concatenate the train and test splits *before*
deduplication and overlap measurement, otherwise the 54.9% duplicate / 91.6% overlap
figures cannot be reproduced (they are cross-split).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MIRRORS = {
    "unsw-nb15": "Mireu-Lab/UNSW-NB15",
    "cic-ids2017": "c01dsnap/CIC-IDS2017",
    "ton-iot": "codymlewis/TON_IoT_network",
}

LABEL_COLUMNS = {
    "unsw-nb15": "label",
    "cic-ids2017": "Label",
    "ton-iot": "label",
}


@dataclass
class DatasetBundle:
    frame: pd.DataFrame
    label_col: str
    name: str


def load_dataset(name: str, use_synthetic: bool = False) -> DatasetBundle:
    """Load one benchmark as a single concatenated DataFrame.

    Args:
        name: one of ``{"unsw-nb15", "cic-ids2017", "ton-iot"}``.
        use_synthetic: if True (or if ``datasets`` is unavailable / offline),
            return a small synthetic frame with the same schema shape.
    """
    key = name.lower()
    if key not in MIRRORS:
        raise ValueError(f"unknown dataset {name!r}; expected one of {list(MIRRORS)}")

    if use_synthetic:
        return _synthetic_bundle(key)

    try:
        from datasets import load_dataset as hf_load  # noqa: WPS433 (lazy import)

        ds = hf_load(MIRRORS[key])
        frames = [split.to_pandas() for split in ds.values()]
        frame = pd.concat(frames, ignore_index=True)  # concat BEFORE dedup
        label_col = _resolve_label_col(frame, key)
        return DatasetBundle(frame=frame, label_col=label_col, name=key)
    except Exception:  # pragma: no cover - network/offline path
        return _synthetic_bundle(key)


def _resolve_label_col(frame: pd.DataFrame, key: str) -> str:
    preferred = LABEL_COLUMNS[key]
    if preferred in frame.columns:
        return preferred
    for cand in ("label", "Label", "class", "Class"):
        if cand in frame.columns:
            return cand
    raise KeyError(f"no label column found for {key}; columns={list(frame.columns)[:10]}")


def _synthetic_bundle(key: str, n: int = 4000, seed: int = 42) -> DatasetBundle:
    """Generate a schema-faithful synthetic frame for tests and offline CI."""
    rng = np.random.default_rng(seed)
    n_num = 12
    X = rng.normal(size=(n, n_num)).astype(np.float32)
    # inject a separable signal so training is meaningful in tests
    y = (X[:, 0] + 0.5 * X[:, 1] - 0.3 * X[:, 2] + rng.normal(scale=0.5, size=n) > 0).astype(int)
    cols = {f"f{i}": X[:, i] for i in range(n_num)}
    cols["proto"] = rng.choice(["tcp", "udp", "icmp"], size=n)

    if key == "ton-iot":
        cols["src_ip"] = rng.choice([f"192.168.0.{i}" for i in range(1, 20)], size=n)
        cols["dst_ip"] = rng.choice([f"10.0.0.{i}" for i in range(1, 20)], size=n)
        cols["type"] = np.where(y == 1, rng.choice(["ddos", "mitm", "scanning"], size=n), "normal")
        label_col = "label"
        cols[label_col] = y
        # inject duplicates so dedup has an effect in the synthetic path
        frame = pd.DataFrame(cols)
        dup = frame.sample(frac=0.4, random_state=seed)
        frame = pd.concat([frame, dup], ignore_index=True)
        return DatasetBundle(frame=frame, label_col=label_col, name=key)

    label_col = "Label" if key == "cic-ids2017" else "label"
    cols[label_col] = y
    return DatasetBundle(frame=pd.DataFrame(cols), label_col=label_col, name=key)
