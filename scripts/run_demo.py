"""End-to-end demo: run the LeakGuard audit on synthetic data in one command.

    python scripts/run_demo.py

Prints each audited factor's mean accuracy so the single-architecture attribution
is visible without downloading any dataset.
"""
from __future__ import annotations

from leakguard.audit import run_audit


def main() -> None:
    report = run_audit("ton-iot", seeds=2, synthetic=True)
    print(f"Dataset: {report['dataset']} (synthetic)\n")
    print(f"{'factor':<24}{'accuracy (mean)':>16}")
    print("-" * 40)
    for name, stats in report["factors"].items():
        acc = stats.get("accuracy_mean")
        if acc is not None:
            print(f"{name:<24}{acc:>16.4f}")
    print("-" * 40)
    for name, m in report["baselines"].items():
        print(f"{name:<24}{m['accuracy']:>16.4f}")


if __name__ == "__main__":
    main()
