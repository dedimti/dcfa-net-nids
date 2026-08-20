"""Training loop for the DCFA-Net probe.

Supports the two checkpoint-selection policies audited in Section 5.2 (Table 7):

    * ``"best"``        -> select the epoch with best validation accuracy
    * ``"last"``        -> use the final-epoch weights unconditionally

The manuscript shows that on TON_IoT, last-epoch selection is not systematically
better or worse but is far noisier across seeds (99.17% vs 96.33 +/- 5.23%). This
loop reproduces that behaviour by tracking both checkpoints.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .metrics import compute_metrics
from .model import build_model


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-5
    d_model: int = 64
    n_heads: int = 4
    hidden: int = 128
    dropout: float = 0.1
    variant: str = "full"
    checkpoint: str = "best"  # "best" | "last"
    val_fraction: float = 0.15
    seed: int = 42
    device: str = "cpu"


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_probe(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: TrainConfig | None = None,
) -> dict[str, float]:
    """Train DCFA-Net and evaluate under the configured checkpoint policy."""
    cfg = config or TrainConfig()
    _set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")

    # carve a validation split for checkpoint selection
    n = len(X_train)
    idx = np.random.default_rng(cfg.seed).permutation(n)
    n_val = max(1, int(n * cfg.val_fraction))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    Xtr = torch.tensor(X_train[tr_idx], dtype=torch.float32)
    ytr = torch.tensor(y_train[tr_idx], dtype=torch.long)
    Xva = torch.tensor(X_train[val_idx], dtype=torch.float32, device=device)
    yva = y_train[val_idx]
    Xte = torch.tensor(X_test, dtype=torch.float32, device=device)

    loader = DataLoader(
        TensorDataset(Xtr, ytr), batch_size=cfg.batch_size, shuffle=True
    )

    model = build_model(
        n_features=X_train.shape[1],
        n_classes=int(max(y_train.max(), y_test.max())) + 1,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        hidden=cfg.hidden,
        dropout=cfg.dropout,
        variant=cfg.variant,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = torch.nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_state = copy.deepcopy(model.state_dict())

    for _ in range(cfg.epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

        val_acc = _eval_acc(model, Xva, yva)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

    if cfg.checkpoint == "best":
        model.load_state_dict(best_state)
    # else: keep final-epoch weights already in `model`

    model.eval()
    with torch.no_grad():
        y_pred = model(Xte).argmax(dim=1).cpu().numpy()
    metrics = compute_metrics(y_test, y_pred)
    metrics["best_val_acc"] = float(best_val_acc)
    metrics["checkpoint"] = cfg.checkpoint  # type: ignore[assignment]
    return metrics


def _eval_acc(model: torch.nn.Module, X: torch.Tensor, y: np.ndarray) -> float:
    model.eval()
    with torch.no_grad():
        pred = model(X).argmax(dim=1).cpu().numpy()
    return float((pred == y).mean())
