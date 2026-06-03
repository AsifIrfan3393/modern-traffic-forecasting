"""Sliding-window traffic dataset with graph and edge-feature loading support."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.preprocessing import StandardScaler, augment_features, ensure_traffic_shape


@dataclass
class TrafficDataBundle:
    train: "TrafficForecastDataset"
    val: "TrafficForecastDataset"
    test: "TrafficForecastDataset"
    adjacency: Optional[torch.Tensor]
    edge_index: Optional[torch.Tensor]
    edge_attr: Optional[torch.Tensor]
    scaler: StandardScaler
    num_nodes: int
    input_dim: int


class TrafficForecastDataset(Dataset):
    """Return tensors shaped x=[window,nodes,features], y=[horizon,nodes]."""

    def __init__(self, features: np.ndarray, targets: np.ndarray, window_size: int, horizon: int) -> None:
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)
        self.window_size = int(window_size)
        self.horizon = int(horizon)
        self.length = max(0, self.features.shape[0] - self.window_size - self.horizon + 1)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.features[index : index + self.window_size]
        y = self.targets[index + self.window_size : index + self.window_size + self.horizon]
        return x, y


def load_array(path: str) -> np.ndarray:
    """Load .npy, .npz, .csv, or .h5/.hdf data into a NumPy array."""

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
    if file_path.suffix == ".npy":
        return np.load(file_path)
    if file_path.suffix == ".npz":
        archive = np.load(file_path)
        key = "data" if "data" in archive else archive.files[0]
        return archive[key]
    if file_path.suffix in {".csv", ".txt"}:
        return pd.read_csv(file_path, header=None).values
    if file_path.suffix in {".h5", ".hdf", ".hdf5"}:
        return pd.read_hdf(file_path).values
    raise ValueError(f"Unsupported data format: {file_path.suffix}")


def load_adjacency(path: Optional[str], num_nodes: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Load an adjacency matrix and convert it to PyG edge_index."""

    if not path:
        return None, None
    adjacency = np.asarray(load_array(path), dtype=np.float32)
    if adjacency.shape != (num_nodes, num_nodes):
        raise ValueError(f"Adjacency shape {adjacency.shape} does not match ({num_nodes}, {num_nodes}).")
    rows, cols = np.nonzero(adjacency > 0)
    edge_index = torch.as_tensor(np.stack([rows, cols], axis=0), dtype=torch.long)
    return torch.as_tensor(adjacency, dtype=torch.float32), edge_index


def load_edge_features(path: Optional[str]) -> Optional[torch.Tensor]:
    if not path:
        return None
    return torch.as_tensor(load_array(path), dtype=torch.float32)


def split_time_series(features: np.ndarray, ratios: Tuple[float, float, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_ratio, val_ratio, _ = ratios
    n = features.shape[0]
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    return features[:train_end], features[train_end:val_end], features[val_end:]


def build_datasets(config: Dict) -> TrafficDataBundle:
    """Build train/validation/test datasets from YAML configuration."""

    data_cfg = config["data"]
    model_cfg = config["model"]
    raw = ensure_traffic_shape(load_array(data_cfg["data_path"]))
    target_feature = int(data_cfg.get("target_feature", 0))
    targets = raw[..., target_feature]
    features, scaler = augment_features(
        raw,
        add_time_features=bool(data_cfg.get("add_time_features", True)),
        add_missing_mask=bool(data_cfg.get("add_missing_mask", True)),
        frequency_minutes=int(data_cfg.get("frequency_minutes", 5)),
        holiday_dates=data_cfg.get("holiday_dates", []),
    )

    ratios = (float(data_cfg.get("train_ratio", 0.7)), float(data_cfg.get("val_ratio", 0.1)), float(data_cfg.get("test_ratio", 0.2)))
    feature_splits = split_time_series(features, ratios)
    target_splits = split_time_series(targets, ratios)

    window_size = int(model_cfg["window_size"])
    horizon = int(model_cfg["prediction_horizon"])
    train = TrafficForecastDataset(feature_splits[0], target_splits[0], window_size, horizon)
    val = TrafficForecastDataset(feature_splits[1], target_splits[1], window_size, horizon)
    test = TrafficForecastDataset(feature_splits[2], target_splits[2], window_size, horizon)
    adjacency, edge_index = load_adjacency(data_cfg.get("adjacency_path"), raw.shape[1])
    edge_attr = load_edge_features(data_cfg.get("edge_features_path"))

    return TrafficDataBundle(train, val, test, adjacency, edge_index, edge_attr, scaler, raw.shape[1], features.shape[-1])