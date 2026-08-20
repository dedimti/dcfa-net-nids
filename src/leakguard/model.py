"""DCFA-Net: Dual Cross-Feature Attention Network.

Fixed *measurement probe* used throughout the LeakGuard audit system, exactly as
described in Section 3 of the manuscript. The architecture is held constant across
every ablation so that any change in accuracy is attributable to the varied
evaluation factor rather than to the model.

Architecture (four stages):
    (i)   input projection                              -> Eq. (1)
    (ii)  dual-branch block (local MLP + global attn)   -> Eq. (2), (3)
    (iii) cross-feature attention fusion                -> Eq. (4)
    (iv)  classification head                           -> Eq. (5)

The three architectural variants used in the Section 5.4 ablation are selected via
the ``variant`` argument: ``"full"`` (both branches fused), ``"local"`` (local MLP
branch only), and ``"global"`` (global self-attention branch only).

Note on framing: DCFA-Net is *not* proposed as a competitive detector. Section 5.4
reports its own ablation as a negative result (local-only matches full fusion on
two of three benchmarks), and a same-pipeline random forest exceeds it on all three
datasets. This is by design: a valid measurement instrument must earn no special
standing as a detector.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LocalBranch(nn.Module):
    """Local feature encoder: a per-feature multi-layer perceptron (Eq. 2)."""

    def __init__(self, d_model: int, hidden: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.net(x))


class GlobalBranch(nn.Module):
    """Global feature encoder: multi-head self-attention over features (Eq. 3)."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        return self.norm(x + attn_out)


class CrossFeatureAttention(nn.Module):
    """Cross-feature attention fusion of the two branches (Eq. 4).

    The local representation attends to the global representation, letting the model
    weight globally-contextualised features by their local salience before fusion.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.cross = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.gate = nn.Linear(2 * d_model, d_model)

    def forward(self, local: torch.Tensor, glob: torch.Tensor) -> torch.Tensor:
        fused, _ = self.cross(local, glob, glob, need_weights=False)
        gate = torch.sigmoid(self.gate(torch.cat([local, glob], dim=-1)))
        return self.norm(gate * fused + (1.0 - gate) * local)


class DCFANet(nn.Module):
    """Dual Cross-Feature Attention Network.

    Args:
        n_features: number of input (tabular) features.
        n_classes: number of output classes (2 for binary attack/benign).
        d_model: per-feature embedding width.
        n_heads: attention heads in the global and cross-feature modules.
        hidden: hidden width of the local MLP branch.
        dropout: dropout probability shared across modules.
        variant: one of ``{"full", "local", "global"}`` (Section 5.4 ablation).
    """

    VARIANTS = ("full", "local", "global")

    def __init__(
        self,
        n_features: int,
        n_classes: int = 2,
        d_model: int = 64,
        n_heads: int = 4,
        hidden: int = 128,
        dropout: float = 0.1,
        variant: str = "full",
    ) -> None:
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f"variant must be one of {self.VARIANTS}, got {variant!r}")
        self.variant = variant
        self.n_features = n_features
        self.d_model = d_model

        # (i) input projection: each scalar feature -> d_model embedding (Eq. 1)
        self.feature_embed = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        self.input_norm = nn.LayerNorm(d_model)

        # (ii) dual branches
        self.local_branch = LocalBranch(d_model, hidden, dropout)
        self.global_branch = GlobalBranch(d_model, n_heads, dropout)

        # (iii) fusion
        self.fusion = CrossFeatureAttention(d_model, n_heads, dropout)

        # (iv) classification head (Eq. 5)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features) -> (batch, n_features, d_model)
        tokens = x.unsqueeze(-1) * self.feature_embed.unsqueeze(0)
        tokens = self.input_norm(tokens)

        if self.variant == "local":
            rep = self.local_branch(tokens)
        elif self.variant == "global":
            rep = self.global_branch(tokens)
        else:  # full
            local = self.local_branch(tokens)
            glob = self.global_branch(tokens)
            rep = self.fusion(local, glob)

        pooled = rep.mean(dim=1)  # mean-pool over feature tokens
        return self.head(pooled)


def build_model(n_features: int, n_classes: int = 2, **kwargs) -> DCFANet:
    """Convenience factory used by the training loop and audit runner."""
    return DCFANet(n_features=n_features, n_classes=n_classes, **kwargs)
