from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.traffic_dataset import TrafficDataBundle
from models.predictor import TrafficForecastingModel
from utils.metrics import calculate_metrics
from utils.visualization import plot_attention_heatmaps, plot_history, plot_node_embeddings, plot_predictions


def build_loss(name: str) -> nn.Module:
    name = name.lower()
    if name == "mae":
        return nn.L1Loss()
    if name == "mse":
        return nn.MSELoss()
    if name == "huber":
        return nn.HuberLoss(delta=1.0)
    raise ValueError(f"Unsupported loss function: {name}")


class WarmupCosineScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, warmup_epochs: int, max_epochs: int, base_lr: float) -> None:
        self.optimizer = optimizer
        self.warmup_epochs = max(1, warmup_epochs)
        self.max_epochs = max(self.warmup_epochs + 1, max_epochs)
        self.base_lr = base_lr

    def step(self, epoch: int) -> None:
        if epoch <= self.warmup_epochs:
            lr = self.base_lr * epoch / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / max(1, self.max_epochs - self.warmup_epochs)
            lr = 0.5 * self.base_lr * (1.0 + np.cos(np.pi * progress))
        for group in self.optimizer.param_groups:
            group["lr"] = lr


class TrafficTrainer:
    def __init__(self, model: TrafficForecastingModel, bundle: TrafficDataBundle, config: Dict, device: torch.device) -> None:
        self.model = model.to(device)
        self.bundle = bundle
        self.config = config
        self.device = device
        train_cfg = config["training"]
        self.criterion = build_loss(train_cfg.get("loss_function", "huber"))
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=float(train_cfg["learning_rate"]),
            weight_decay=float(train_cfg["weight_decay"]),
        )
        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            int(train_cfg.get("warmup_epochs", 5)),
            int(train_cfg["epochs"]),
            float(train_cfg["learning_rate"]),
        )
        self.use_amp = bool(train_cfg.get("mixed_precision", True)) and device.type == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)
        self.history = {"train_loss": [], "val_loss": [], "mae": [], "rmse": []}

    def _loader(self, split: str, shuffle: bool) -> DataLoader:
        train_cfg = self.config["training"]
        dataset = getattr(self.bundle, split)
        return DataLoader(
            dataset,
            batch_size=int(train_cfg["batch_size"]),
            shuffle=shuffle,
            num_workers=int(train_cfg.get("num_workers", 0)),
            pin_memory=bool(train_cfg.get("pin_memory", True)) and self.device.type == "cuda",
        )

    def _graph_inputs(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        adjacency = self.bundle.adjacency.to(self.device) if self.bundle.adjacency is not None else None
        edge_attr = self.bundle.edge_attr.to(self.device) if self.bundle.edge_attr is not None else None
        return adjacency, edge_attr

    def _autocast(self):
        return torch.autocast(device_type="cuda", dtype=torch.float16) if self.use_amp else nullcontext()

    def train_epoch(self, epoch: int) -> float:
        self.model.train()
        self.scheduler.step(epoch)
        adjacency, edge_attr = self._graph_inputs()
        loader = self._loader("train", shuffle=True)
        total = 0.0
        for x, y in tqdm(loader, desc=f"train {epoch}", leave=False):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with self._autocast():
                pred = self.model(x, adjacency, edge_attr)
                loss = self.criterion(pred, y)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(self.config["training"].get("gradient_clip_norm", 5.0)))
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total += float(loss.detach().cpu()) * x.shape[0]
        return total / max(1, len(loader.dataset))

    @torch.no_grad()
    def evaluate_split(self, split: str = "val") -> Tuple[float, np.ndarray, np.ndarray]:
        self.model.eval()
        adjacency, edge_attr = self._graph_inputs()
        loader = self._loader(split, shuffle=False)
        losses = []
        predictions = []
        targets = []
        for x, y in tqdm(loader, desc=split, leave=False):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            pred = self.model(x, adjacency, edge_attr)
            losses.append(float(self.criterion(pred, y).detach().cpu()) * x.shape[0])
            predictions.append(pred.detach().cpu().numpy())
            targets.append(y.detach().cpu().numpy())
        if not predictions:
            return float("inf"), np.empty((0,)), np.empty((0,))
        return sum(losses) / max(1, len(loader.dataset)), np.concatenate(predictions), np.concatenate(targets)

    def fit(self) -> Dict[str, float]:
        output_cfg = self.config["outputs"]
        checkpoint_dir = Path(output_cfg["checkpoint_dir"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best_loss = float("inf")
        patience = int(self.config["training"].get("patience", 10))
        wait = 0

        for epoch in range(1, int(self.config["training"]["epochs"]) + 1):
            train_loss = self.train_epoch(epoch)
            val_loss, val_pred, val_true = self.evaluate_split("val")
            metrics = calculate_metrics(val_true, val_pred) if val_pred.size else None
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["mae"].append(metrics.mae if metrics else float("inf"))
            self.history["rmse"].append(metrics.rmse if metrics else float("inf"))
            print(f"epoch={epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} mae={self.history['mae'][-1]:.4f}")

            if val_loss < best_loss:
                best_loss = val_loss
                wait = 0
                torch.save({"model": self.model.state_dict(), "config": self.config}, checkpoint_dir / "best_model.pt")
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stopping after {epoch} epochs.")
                    break

        plot_history(self.history, output_cfg["figure_dir"])
        _, test_pred, test_true = self.evaluate_split("test")
        report = calculate_metrics(test_true, test_pred).to_dict() if test_pred.size else {}
        Path(output_cfg["report_dir"]).mkdir(parents=True, exist_ok=True)
        Path(output_cfg["report_dir"], "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        if test_pred.size:
            plot_predictions(test_true, test_pred, output_cfg["figure_dir"])
        maps = {key: value.detach().cpu().numpy() for key, value in self.model.attention_maps().items()}
        plot_attention_heatmaps(maps, output_cfg["figure_dir"])
        plot_node_embeddings(self.model.node_embeddings().detach().cpu().numpy(), output_cfg["figure_dir"])
        return report