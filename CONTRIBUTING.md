# Contributing

Thanks for your interest in LeakGuard. This repository is the reference
implementation for an academic paper, so contributions that improve
reproducibility, documentation, or test coverage are especially welcome.

## Development setup

```bash
git clone https://github.com/dedimti/dcfa-net-nids.git
cd dcfa-net-nids
pip install -e ".[dev]"
pytest -q          # run the test suite
ruff check .       # lint
```

## Guidelines

- Keep the DCFA-Net probe fixed. It is a *measurement instrument*, not a
  detector to be tuned — changing its capacity would break the single-architecture
  attribution the audit relies on.
- Every new audited factor should be a single toggle that varies one thing at a
  time, and negative / null results should be reported rather than dropped.
- Add or update a unit test for any behavioural change.
- Run `ruff check .` and `pytest -q` before opening a pull request.

## Reporting issues

Please include the dataset, the exact command, and the full traceback.
