"""Generate a small synthetic NIDS-shaped dataset to CSV for offline demos.

Usage:
    python scripts/generate_synthetic.py --dataset ton-iot --out data/ton_synth.csv
"""
from __future__ import annotations

import argparse

from leakguard.data import load_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ton-iot", choices=["unsw-nb15", "cic-ids2017", "ton-iot"])
    ap.add_argument("--out", default="synthetic.csv")
    args = ap.parse_args()

    bundle = load_dataset(args.dataset, use_synthetic=True)
    bundle.frame.to_csv(args.out, index=False)
    print(f"wrote {len(bundle.frame)} rows to {args.out} (label column: {bundle.label_col})")


if __name__ == "__main__":
    main()
