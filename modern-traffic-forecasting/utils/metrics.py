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
    smape: float
    wape: float
    mase: float
    pearson: float
    nse: float

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
import numpy as np
from sklearn.metrics import mean_squared_error

def smape(y_true, y_pred):
    return 100 * np.mean(
        2 * np.abs(y_pred - y_true) /
        (np.abs(y_true) + np.abs(y_pred) + 1e-8)
    )

def wape(y_true, y_pred):
    return 100 * np.sum(np.abs(y_true - y_pred)) / (
        np.sum(np.abs(y_true)) + 1e-8
    )

def mase(y_true, y_pred):
    naive_error = np.mean(np.abs(np.diff(y_true)))
    model_error = np.mean(np.abs(y_true - y_pred))
    return model_error / (naive_error + 1e-8)

def pearson_corr(y_true, y_pred):
    return np.corrcoef(y_true.flatten(),
                       y_pred.flatten())[0,1]

def nse(y_true, y_pred):
    return 1 - (
        np.sum((y_true - y_pred)**2) /
        np.sum((y_true - np.mean(y_true))**2)
    )