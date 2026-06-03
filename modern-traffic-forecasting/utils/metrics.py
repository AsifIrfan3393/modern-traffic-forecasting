from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict

import numpy as np
from sklearn.metrics import explained_variance_score, mean_absolute_error, r2_score


@dataclass(frozen=True)
class MetricsReport:
    mae: float
    rmse: float
    mape: float
    r2: float
    explained_variance: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def _flatten(y: np.ndarray) -> np.ndarray:
    return np.asarray(y, dtype=np.float64).reshape(-1)


def masked_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-5) -> float:
    """MAPE that ignores zero/near-zero targets to avoid exploding reports."""

    truth = _flatten(y_true)
    pred = _flatten(y_pred)
    mask = np.abs(truth) > eps
    if not np.any(mask):
        return 0.0
    return float(np.mean(np.abs((truth[mask] - pred[mask]) / truth[mask])) * 100.0)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> MetricsReport:
    """Calculate MAE, RMSE, MAPE, R², and explained variance."""

    truth = _flatten(y_true)
    pred = _flatten(y_pred)
    mae = float(mean_absolute_error(truth, pred))
    rmse = float(np.sqrt(np.mean((truth - pred) ** 2)))
    mape = masked_mape(truth, pred)
    r2 = float(r2_score(truth, pred)) if truth.size > 1 else 0.0
    evs = float(explained_variance_score(truth, pred)) if truth.size > 1 else 0.0
    return MetricsReport(mae=mae, rmse=rmse, mape=mape, r2=r2, explained_variance=evs)