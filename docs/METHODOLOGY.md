# Methodology: single-architecture attribution

LeakGuard isolates the effect of each evaluation choice by holding the model fixed
and varying one factor at a time. This document maps each audited factor to the code
and to the manuscript.

## The fixed probe

`DCFANet` (`src/leakguard/model.py`) is a dual-branch tabular model: a local
per-feature MLP branch and a global multi-head self-attention branch, fused by a
cross-feature attention module. It is **not** proposed as a competitive detector.
Section 5.4 reports its own ablation as a negative result — the local branch alone
matches or exceeds the full fusion on two of three benchmarks — and a same-pipeline
random forest exceeds it on all three datasets. A valid measurement instrument must
earn no special standing as a detector.

## Audited factors

| Factor | Code toggle | Manuscript |
|---|---|---|
| Cross-split deduplication | `PreprocessConfig.deduplicate` | Section 4.2, Table 5 |
| Label-determining field guard | `PreprocessConfig.drop_label_determining` | Section 4.2 |
| Identifier (IP) encoding | `PreprocessConfig.encode_identifiers` | Section 5.3 |
| Checkpoint policy | `TrainConfig.checkpoint` (`best`/`last`) | Section 5.2, Table 7 |
| Architectural variant | `TrainConfig.variant` (`full`/`local`/`global`) | Section 5.4, Table 8 |

Each factor is a single toggle so its marginal effect is read on a common scale.

## The central asymmetry (Section 6.1)

On these benchmarks, data-preparation decisions moved the reported score by whole
percentage points, while training-procedure and architecture decisions moved it by
hundredths:

- Removing duplicates changes what 91.6% of the TON_IoT test set even measures.
- Excluding the label-determining `type` field is necessary for the task to be
  non-trivial.
- Identifier encoding is worth a further 0.79 percentage points.
- The full-vs-local accuracy difference never exceeded 0.23 percentage points
  (p ≥ 0.27), indistinguishable from seed variation.

## Reproducibility rules learned the hard way

1. **Concatenate all splits before deduplication.** Cross-split duplicates survive
   otherwise, and the 54.9% / 91.6% TON_IoT figures cannot be reproduced.
2. **Shuffle before subsampling.** These datasets ship class-ordered.
3. **Report nulls.** Factors that move the score by nothing are reported, not dropped.
