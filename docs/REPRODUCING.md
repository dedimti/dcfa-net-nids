# Reproducing the paper results

## Offline (no download) — verify the pipeline

```bash
pip install -e ".[dev]"
python scripts/run_demo.py     # runs the full audit on synthetic data
pytest -q                      # 14 unit tests
```

## With the real benchmarks

Install the optional data dependency and run the audit runner. The loaders read
from the public Hugging Face mirrors named in the paper:

```bash
pip install -e ".[data]"

python -m leakguard.audit --dataset unsw-nb15   --seeds 3 --out results_unsw.json
python -m leakguard.audit --dataset cic-ids2017 --seeds 3 --out results_cic.json
python -m leakguard.audit --dataset ton-iot     --seeds 3 --out results_ton.json
```

| Mirror | Dataset |
|---|---|
| `Mireu-Lab/UNSW-NB15` | UNSW-NB15 |
| `c01dsnap/CIC-IDS2017` | CIC-IDS2017 |
| `codymlewis/TON_IoT_network` | TON_IoT |

## Reference figures (leakage-aware probe, mean over seeds)

| Dataset | DCFA-Net accuracy | F1 | Random forest (same pipeline) |
|---|---|---|---|
| UNSW-NB15 | 97.33 ± 0.20% | 0.9789 | 98.10 ± 0.04% |
| CIC-IDS2017 | 97.40 ± 0.21% | 0.9277 | 99.80 ± 0.02% |
| TON_IoT (IP-dropped) | 99.17 ± 0.09% | 0.9947 | 99.81 ± 0.01% |

The random forest exceeding the probe on every dataset is expected and required —
see `docs/METHODOLOGY.md`.

## Leave-one-attack-out (Section 5.5)

```python
from leakguard import load_dataset, run_loao
from leakguard.train import TrainConfig

bundle = load_dataset("ton-iot")   # or use_synthetic=True
run_loao(bundle.frame, bundle.label_col, ["mitm", "ddos"],
         train_config=TrainConfig(epochs=30))
```

Reference recall on unseen categories: UNSW-NB15 DoS 98.67%, Worms 94.44%;
TON_IoT mitm 38.00%, ddos 84.57%.

> Numbers depend on the exact public-mirror snapshot. The audit *procedure* and the
> relative ordering of factor effects are the reproducible contribution, not any
> single decimal.
