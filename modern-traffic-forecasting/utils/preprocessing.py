from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class StandardScaler:
    """Simple feature-wise scaler for arrays shaped [time, nodes, features]."""

    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None
    eps: float = 1e-6

    def fit(self, data: np.ndarray) -> "StandardScaler":
        self.mean = np.nanmean(data, axis=(0, 1), keepdims=True)
        self.std = np.nanstd(data, axis=(0, 1), keepdims=True)
        self.std = np.where(self.std < self.eps, 1.0, self.std)
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler must be fit before transform().")
        return (data - self.mean) / self.std

    def inverse_transform_target(self, data: np.ndarray, target_feature: int = 0) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("StandardScaler must be fit before inverse_transform_target().")
        return data * self.std[..., target_feature] + self.mean[..., target_feature]


def clean_missing_values(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Replace NaN/Inf values and return a binary observed-value mask."""

    data = np.asarray(data, dtype=np.float32)
    mask = np.isfinite(data).astype(np.float32)
    cleaned = np.where(np.isfinite(data), data, np.nan)
    feature_medians = np.nanmedian(cleaned, axis=(0, 1), keepdims=True)
    feature_medians = np.where(np.isfinite(feature_medians), feature_medians, 0.0)
    cleaned = np.where(np.isnan(cleaned), feature_medians, cleaned)
    return cleaned.astype(np.float32), mask.astype(np.float32)


def ensure_traffic_shape(data: np.ndarray) -> np.ndarray:
    """Normalize common traffic data layouts to [time, nodes, features]."""

    data = np.asarray(data)
    if data.ndim == 2:
        return data[..., None].astype(np.float32)
    if data.ndim != 3:
        raise ValueError("Traffic data must have shape [time,nodes], [time,nodes,features], or [nodes,time,features].")
    # If first dimension is clearly nodes and second is time, transpose to time-major.
    if data.shape[0] < data.shape[1] and data.shape[0] <= 1024:
        return np.transpose(data, (1, 0, 2)).astype(np.float32)
    return data.astype(np.float32)


def build_time_features(
    num_steps: int,
    frequency_minutes: int = 5,
    start_time: Optional[str] = None,
    holiday_dates: Optional[Iterable[str]] = None,
) -> np.ndarray:
    """Create cyclical time-of-day/week/month plus weekend and holiday indicators."""

    start = pd.Timestamp(start_time) if start_time else pd.Timestamp("2020-01-01")
    index = pd.date_range(start=start, periods=num_steps, freq=f"{frequency_minutes}min")
    holidays = {pd.Timestamp(day).date() for day in (holiday_dates or [])}

    minute_of_day = index.hour * 60 + index.minute
    day_of_week = index.dayofweek.to_numpy()
    month = (index.month - 1).to_numpy()

    features = np.stack(
        [
            np.sin(2 * np.pi * minute_of_day / 1440),
            np.cos(2 * np.pi * minute_of_day / 1440),
            np.sin(2 * np.pi * day_of_week / 7),
            np.cos(2 * np.pi * day_of_week / 7),
            np.sin(2 * np.pi * month / 12),
            np.cos(2 * np.pi * month / 12),
            (day_of_week >= 5).astype(np.float32),
            np.array([stamp.date() in holidays for stamp in index], dtype=np.float32),
        ],
        axis=-1,
    )
    return features.astype(np.float32)


def augment_features(
    data: np.ndarray,
    add_time_features: bool = True,
    add_missing_mask: bool = True,
    frequency_minutes: int = 5,
    holiday_dates: Optional[Iterable[str]] = None,
) -> Tuple[np.ndarray, StandardScaler]:
    """Clean, scale, and augment raw traffic features."""

    data = ensure_traffic_shape(data)
    cleaned, mask = clean_missing_values(data)
    scaler = StandardScaler().fit(cleaned)
    scaled = scaler.transform(cleaned).astype(np.float32)

    features = [scaled]
    if add_missing_mask:
        features.append(mask)
    if add_time_features:
        time_features = build_time_features(scaled.shape[0], frequency_minutes, holiday_dates=holiday_dates)
        tiled = np.repeat(time_features[:, None, :], scaled.shape[1], axis=1)
        features.append(tiled)

    return np.concatenate(features, axis=-1).astype(np.float32), scaler