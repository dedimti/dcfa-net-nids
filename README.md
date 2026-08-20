# LeakGuard

**An intelligent audit system that attributes reported NIDS accuracy to evaluation choices rather than detection skill.**

[![CI](https://github.com/dedimti/dcfa-net-nids/actions/workflows/ci.yml/badge.svg)](https://github.com/dedimti/dcfa-net-nids/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)

Reference implementation for the *Expert Systems With Applications* paper
**"LeakGuard: An Intelligent Audit System Attributing Reported NIDS Accuracy to
Evaluation Choices Rather Than Detection Skill."**

Many recent NIDS studies report accuracies above 99% on standard benchmarks — a
regime where the headline number stops telling apart a detector that has learned to
recognise *attacks* from one that has learned to recognise the *dataset*. LeakGuard
answers a single operational question: **how much of a reported score reflects
genuine detection capability, and how much is an artifact of how the evaluation was
set up?**

It does this by **holding the architecture fixed** (the DCFA-Net probe) and
**varying one evaluation factor at a time**, so each factor's marginal effect is
placed on one comparable scale.

## The probe is not a detector

DCFA-Net is a *measurement instrument*, not a proposed detector. On these
benchmarks a same-pipeline random forest exceeds it on all three datasets, and its
own architectural ablation is a null result. This is by design: a valid instrument
must earn no special standing as a detector, so that any accuracy change across an
ablation is read as the effect of the *varied factor*, never as evidence for the
probe itself.

## Quickstart (no download)

```bash
git clone https://github.com/dedimti/dcfa-net-nids.git
cd dcfa-net-nids
pip install -e ".[dev]"

python scripts/run_demo.py     # full audit on synthetic data, one command
pytest -q                      # 14 unit tests
```

## Run the audit on a real benchmark

```bash
pip install -e ".[data]"
python -m leakguard.audit --dataset ton-iot --seeds 3 --out results_ton.json
# datasets: unsw-nb15 | cic-ids2017 | ton-iot
```

The runner varies each factor against the fixed probe and reports its marginal
effect on accuracy, plus same-pipeline random-forest and logistic-regression
baselines.

## What LeakGuard audits

| Factor | Toggle | Paper |
|---|---|---|
| Cross-split deduplication | `PreprocessConfig.deduplicate` | §4.2, Table 5 |
| Label-determining field guard (e.g. TON_IoT `type`) | `PreprocessConfig.drop_label_determining` | §4.2 |
| Identifier (IP) encoding | `PreprocessConfig.encode_identifiers` | §5.3 |
| Checkpoint policy (`best` / `last`) | `TrainConfig.checkpoint` | §5.2, Table 7 |
| Architectural variant (`full` / `local` / `global`) | `TrainConfig.variant` | §5.4, Table 8 |

## The central finding (§6.1)

On these benchmarks, the decisions a reader never sees moved the reported score by
**whole percentage points**, while the decisions every paper foregrounds moved it by
**hundredths**:

- Removing duplicates changes what **91.6%** of the TON_IoT test set even measures
  (54.9% of records are exact duplicates).
- Dropping the label-determining `type` field is necessary for the task to be
  non-trivial at all.
- Identifier encoding is worth a further **0.79** percentage points.
- The full-vs-local accuracy difference never exceeded **0.23** percentage points
  (p ≥ 0.27) — indistinguishable from seed variation.

## Reference results (leakage-aware probe, mean ± std over seeds)

| Dataset | DCFA-Net | F1 | Random forest (same pipeline) |
|---|---|---|---|
| UNSW-NB15 | 97.33 ± 0.20% | 0.9789 | 98.10 ± 0.04% |
| CIC-IDS2017 | 97.40 ± 0.21% | 0.9277 | 99.80 ± 0.02% |
| TON_IoT (IP-dropped) | 99.17 ± 0.09% | 0.9947 | 99.81 ± 0.01% |

Leave-one-attack-out recall on unseen categories (§5.5): UNSW-NB15 DoS 98.67%,
Worms 94.44%; TON_IoT mitm 38.00%, ddos 84.57%.

## Repository layout

```
src/leakguard/
  model.py          DCFA-Net probe (full / local / global variants)
  preprocessing.py  leakage-aware controls (dedup, label-guard, IP handling)
  data.py           public-mirror loaders + offline synthetic fallback
  train.py          training loop with best / last checkpoint selection
  baselines.py      same-pipeline random forest & logistic regression
  metrics.py        accuracy, attack P/R/F1, FPR, seed aggregation
  loao.py           leave-one-attack-out zero-day protocol
  audit.py          one-command audit runner (CLI: leakguard-audit)
configs/            per-dataset audit configurations
scripts/            run_demo.py, generate_synthetic.py
tests/              14 unit tests
docs/               METHODOLOGY.md, REPRODUCING.md
```

## Data sources

Public Hugging Face mirrors: `Mireu-Lab/UNSW-NB15`, `c01dsnap/CIC-IDS2017`,
`codymlewis/TON_IoT_network`. See [`docs/REPRODUCING.md`](docs/REPRODUCING.md).

## Citation

If you use this software, please cite the paper and the software (see
[`CITATION.cff`](CITATION.cff)):

```
Irawan, D., Sudarmaji, & Samsudin, I. (2026). LeakGuard: An Intelligent Audit
System Attributing Reported NIDS Accuracy to Evaluation Choices Rather Than
Detection Skill. Expert Systems With Applications.
```

## License

MIT — see [LICENSE](LICENSE).
