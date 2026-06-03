"""Evaluate a saved checkpoint on the configured test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from dataset.traffic_dataset import build_datasets
from models.predictor import TrafficForecastingModel
from trainers.trainer import TrafficTrainer
from train import choose_device
from utils.metrics import calculate_metrics
from utils.seed import set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate traffic forecasting checkpoint.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best_model.pt")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    set_seed(int(config.get("seed", 42)))
    device = choose_device(config.get("device", "auto"))
    bundle = build_datasets(config)
    model = TrafficForecastingModel(bundle.num_nodes, bundle.input_dim, config)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    trainer = TrafficTrainer(model, bundle, config, device)
    _, pred, truth = trainer.evaluate_split("test")

import numpy as np

np.save("test_pred.npy", pred)
np.save("test_true.npy", truth)

report = calculate_metrics(truth, pred).to_dict()

print(json.dumps(report, indent=2))
print("Saved test_pred.npy")
print("Saved test_true.npy")


if __name__ == "__main__":
    main()