"""Leakage-aware preprocessing for tabular NIDS benchmarks.

This module implements the data-preparation controls audited in Section 4.2 of the
manuscript. Each control is exposed as an explicit, toggleable step so the audit
runner can vary *one factor at a time* and attribute its marginal effect on a
common scale:

    * cross-split deduplication      (Section 4.2, Table 5)
    * label-determining-field guard  (e.g. TON_IoT ``type``)
    * identifier-column handling      (e.g. ``src_ip`` / ``dst_ip``)
    * categorical / numeric encoding

Design rule enforced here (learned the hard way): concatenate *all* splits before
deduplication, otherwise cross-split duplicates survive; and shuffle before any
subsampling, because these datasets ship class-ordered.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler


@dataclass
class PreprocessConfig:
    """Toggles for the data-preparation factors audited in Section 4.2."""

    deduplicate: bool = True
    """Remove exact-duplicate rows across the *concatenated* splits before splitting."""

    drop_label_determining: bool = True
    """Drop fields that trivially determine the label (e.g. TON_IoT ``type``)."""

    encode_identifiers: bool = False
    """If False, drop identifier columns (IP/port); if True, label-encode them.

    The IP-leakage effect in Section 5.3 is measured by flipping this toggle."""

    identifier_columns: list[str] = field(
        default_factory=lambda: ["src_ip", "dst_ip", "srcip", "dstip", "src_port", "dst_port"]
    )
    label_determining_columns: list[str] = field(
        default_factory=lambda: ["type", "attack_cat", "attack_category"]
    )
    random_state: int = 42


class LeakageAwarePreprocessor:
    """Fit/transform preprocessor that applies the configured leakage controls."""

    def __init__(self, config: PreprocessConfig | None = None) -> None:
        self.config = config or PreprocessConfig()
        self.scaler = StandardScaler()
        self.cat_encoders: dict[str, LabelEncoder] = {}
        self.feature_names_: list[str] = []
        self._fitted = False

    # -- public API ---------------------------------------------------------

    def deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop exact-duplicate rows. Caller must pass *all* splits concatenated."""
        if not self.config.deduplicate:
            return df
        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        self.n_duplicates_removed_ = before - len(df)
        self.duplicate_fraction_ = self.n_duplicates_removed_ / max(before, 1)
        return df

    def fit_transform(
        self, df: pd.DataFrame, label_col: str
    ) -> tuple[np.ndarray, np.ndarray]:
        df = self._guard_and_drop(df, label_col)
        y = self._encode_label(df[label_col].to_numpy())
        X_df = df.drop(columns=[label_col])
        X_df = self._encode_features(X_df, fit=True)
        X = self.scaler.fit_transform(X_df.to_numpy(dtype=np.float32))
        self.feature_names_ = list(X_df.columns)
        self._fitted = True
        return X.astype(np.float32), y

    def transform(self, df: pd.DataFrame, label_col: str) -> tuple[np.ndarray, np.ndarray]:
        if not self._fitted:
            raise RuntimeError("call fit_transform before transform")
        df = self._guard_and_drop(df, label_col)
        y = self._encode_label(df[label_col].to_numpy())
        X_df = df.drop(columns=[label_col])
        X_df = self._encode_features(X_df, fit=False)
        X_df = X_df.reindex(columns=self.feature_names_, fill_value=0.0)
        X = self.scaler.transform(X_df.to_numpy(dtype=np.float32))
        return X.astype(np.float32), y

    # -- internals ----------------------------------------------------------

    def _guard_and_drop(self, df: pd.DataFrame, label_col: str) -> pd.DataFrame:
        df = df.copy()
        drop_cols: list[str] = []
        if self.config.drop_label_determining:
            drop_cols += [
                c
                for c in self.config.label_determining_columns
                if c in df.columns and c != label_col
            ]
        if not self.config.encode_identifiers:
            drop_cols += [c for c in self.config.identifier_columns if c in df.columns]
        return df.drop(columns=list(dict.fromkeys(drop_cols)), errors="ignore")

    def _encode_label(self, y: np.ndarray) -> np.ndarray:
        # Binary task: any non-zero / non-"normal" label is an attack.
        if y.dtype.kind in "iufb":
            return (y != 0).astype(np.int64)
        y_str = np.asarray([str(v).strip().lower() for v in y])
        benign = {"0", "normal", "benign", "background"}
        return np.array([0 if v in benign else 1 for v in y_str], dtype=np.int64)

    def _encode_features(self, X_df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        X_df = X_df.copy()
        for col in X_df.columns:
            if X_df[col].dtype == object or str(X_df[col].dtype).startswith("category"):
                if fit:
                    enc = LabelEncoder()
                    X_df[col] = enc.fit_transform(X_df[col].astype(str))
                    self.cat_encoders[col] = enc
                else:
                    enc = self.cat_encoders.get(col)
                    if enc is None:
                        X_df[col] = 0
                    else:
                        known = set(enc.classes_)
                        X_df[col] = [
                            enc.transform([v])[0] if v in known else 0
                            for v in X_df[col].astype(str)
                        ]
        return X_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)


def measure_train_test_overlap(
    train: pd.DataFrame, test: pd.DataFrame
) -> dict[str, float]:
    """Fraction of test rows that also appear in train (Section 4.2, Table 5).

    Used to reproduce the TON_IoT figure: 91.6% of test rows also appear in
    training before deduplication.
    """
    train_keys = set(map(tuple, train.to_numpy().tolist()))
    hits = sum(1 for row in test.to_numpy().tolist() if tuple(row) in train_keys)
    return {
        "test_rows": float(len(test)),
        "overlapping_rows": float(hits),
        "overlap_fraction": hits / max(len(test), 1),
    }
