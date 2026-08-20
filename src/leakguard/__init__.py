"""LeakGuard: an intelligent audit system for tabular NIDS benchmarks.

Reference implementation accompanying the Expert Systems With Applications paper
"LeakGuard: An Intelligent Audit System Attributing Reported NIDS Accuracy to
Evaluation Choices Rather Than Detection Skill".

The DCFA-Net model is a fixed *measurement probe*, not a proposed detector.
"""
from .baselines import run_baseline
from .data import DatasetBundle, load_dataset
from .loao import run_loao
from .metrics import aggregate_seeds, compute_metrics
from .model import DCFANet, build_model
from .preprocessing import (
    LeakageAwarePreprocessor,
    PreprocessConfig,
    measure_train_test_overlap,
)
from .train import TrainConfig, train_probe

__version__ = "1.0.0"

__all__ = [
    "DCFANet",
    "build_model",
    "LeakageAwarePreprocessor",
    "PreprocessConfig",
    "measure_train_test_overlap",
    "load_dataset",
    "DatasetBundle",
    "compute_metrics",
    "aggregate_seeds",
    "run_baseline",
    "train_probe",
    "TrainConfig",
    "run_loao",
    "__version__",
]
