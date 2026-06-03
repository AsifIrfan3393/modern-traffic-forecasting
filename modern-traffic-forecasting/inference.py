"""Run single-window inference with a trained traffic forecasting checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from dataset.traffic_dataset import build_datasets
from models.predictor import TrafficForecastingModel
from train import choose_device


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Forecast traffic for the first available test window.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/best_model.pt")
    parser.add_argument("--output", default="outputs/reports/inference_prediction.json")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    device = choose_device(config.get("device", "auto"))
    bundle = build_datasets(config)
    model = TrafficForecastingModel(bundle.num_nodes, bundle.input_dim, config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    x, _ = bundle.test[0]
    adjacency = bundle.adjacency.to(device) if bundle.adjacency is not None else None
    edge_attr = bundle.edge_attr.to(device) if bundle.edge_attr is not None else None
    pred = model(x.unsqueeze(0).to(device), adjacency, edge_attr).squeeze(0).cpu().numpy()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"prediction": np.asarray(pred).tolist()}, indent=2), encoding="utf-8")
    print(f"Saved forecast to {args.output}")


if __name__ == "__main__":
    main()