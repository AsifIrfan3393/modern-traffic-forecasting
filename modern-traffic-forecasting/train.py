from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch
import yaml

from dataset.traffic_dataset import build_datasets
from models.predictor import TrafficForecastingModel
from trainers.trainer import TrafficTrainer
from utils.seed import set_seed


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train adaptive GATv2-TFT traffic forecaster.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config.")
    args = parser.parse_args()
    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    device = choose_device(config.get("device", "auto"))
    bundle = build_datasets(config)
    model = TrafficForecastingModel(bundle.num_nodes, bundle.input_dim, config)
    trainer = TrafficTrainer(model, bundle, config, device)
    report = trainer.fit()
    print(report)


if __name__ == "__main__":
    main()