"""Generate publication-style figures from trained traffic forecasting checkpoints.

Examples:
    python plot_results.py --config configs/metr_la.yaml \
        --checkpoint outputs/METR-LA/checkpoints/best_model.pt \
        --output-dir outputs/METR-LA/research_figures

    python plot_results.py --compare-json METR-LA=outputs/METR-LA/reports/metrics.json \
        --compare-json PeMSD8=outputs/PEMSD8/reports/metrics.json \
        --output-dir outputs/comparison_figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml
from sklearn.manifold import TSNE

from dataset.traffic_dataset import build_datasets
from models.predictor import TrafficForecastingModel
from train import choose_device
from trainers.trainer import TrafficTrainer
from utils.metrics import calculate_metrics
from utils.seed import set_seed


FIGURE_DPI = 120


def save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)


def horizon_indices(horizon: int, frequency_minutes: int) -> List[Tuple[int, int]]:
    desired_minutes = [15, 30, 45, 60]
    indices: List[Tuple[int, int]] = []
    for minutes in desired_minutes:
        idx = min(horizon - 1, max(0, int(round(minutes / frequency_minutes)) - 1))
        if (idx, minutes) not in indices:
            indices.append((idx, minutes))
    return indices


def collect_predictions(config: Dict, checkpoint_path: str) -> Tuple[np.ndarray, np.ndarray, TrafficForecastingModel]:
    set_seed(int(config.get("seed", 42)))
    device = choose_device(config.get("device", "auto"))
    bundle = build_datasets(config)
    model = TrafficForecastingModel(bundle.num_nodes, bundle.input_dim, config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    trainer = TrafficTrainer(model, bundle, config, device)
    _, predictions, targets = trainer.evaluate_split("test")
    return predictions, targets, trainer.model


def plot_horizon_grid(
    targets: np.ndarray,
    predictions: np.ndarray,
    output_dir: Path,
    frequency_minutes: int,
    dataset_name: str,
    node_index: int = 0,
    max_points: int = 300,
) -> None:
    horizon = predictions.shape[1]
    panels = horizon_indices(horizon, frequency_minutes)
    fig, axes = plt.subplots(2, 2, figsize=(8, 5.5), sharex=False)
    axes = axes.reshape(-1)
    panel_letters = ["(a)", "(b)", "(c)", "(d)"]

    for axis, (h_idx, minutes), letter in zip(axes, panels, panel_letters):
        actual = targets[:max_points, h_idx, node_index]
        pred = predictions[:max_points, h_idx, node_index]
        axis.plot(actual, label="Actual", linewidth=1.2, color="#1f77b4")
        axis.plot(pred, label="Predicted", linewidth=1.2, color="#ff7f0e")
        axis.set_title(f"{letter} {minutes} minutes prediction", fontsize=11, fontweight="bold")
        axis.set_xlabel("Test window")
        axis.set_ylabel("Traffic value")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)

    for axis in axes[len(panels) :]:
        axis.axis("off")

    fig.suptitle(f"{dataset_name}: Multi-horizon Forecast vs Actual", fontsize=12, fontweight="bold")
    save_figure(fig, output_dir / "multi_horizon_prediction_grid.png")


def plot_per_horizon_metrics(targets: np.ndarray, predictions: np.ndarray, output_dir: Path, frequency_minutes: int) -> None:
    maes, rmses, mapes, r2s = [], [], [], []
    horizon = predictions.shape[1]
    minutes = np.arange(1, horizon + 1) * frequency_minutes
    for h_idx in range(horizon):
        report = calculate_metrics(targets[:, h_idx, :], predictions[:, h_idx, :])
        maes.append(report.mae)
        rmses.append(report.rmse)
        mapes.append(report.mape)
        r2s.append(report.r2)

    fig, axes = plt.subplots(2, 2, figsize=(8, 5.5))
    for axis, values, title, ylabel in [
        (axes[0, 0], maes, "MAE by Forecast Horizon", "MAE"),
        (axes[0, 1], rmses, "RMSE by Forecast Horizon", "RMSE"),
        (axes[1, 0], mapes, "MAPE by Forecast Horizon", "MAPE (%)"),
        (axes[1, 1], r2s, "R² by Forecast Horizon", "R²"),
    ]:
        axis.plot(minutes, values, marker="o", linewidth=1.6)
        axis.set_title(title, fontweight="bold")
        axis.set_xlabel("Forecast horizon (minutes)")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
    save_figure(fig, output_dir / "per_horizon_metrics.png")


def plot_error_diagnostics(targets: np.ndarray, predictions: np.ndarray, output_dir: Path) -> None:
    truth = targets.reshape(-1)
    pred = predictions.reshape(-1)
    residuals = truth - pred
    limit = min(2000, truth.size)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(truth[:limit], pred[:limit], s=8, alpha=0.35)
    min_val = min(float(truth[:limit].min()), float(pred[:limit].min()))
    max_val = max(float(truth[:limit].max()), float(pred[:limit].max()))
    ax.plot([min_val, max_val], [min_val, max_val], color="black", linestyle="--", linewidth=1)
    ax.set_title("Predicted vs Actual Scatter", fontweight="bold")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.grid(True, alpha=0.25)
    save_figure(fig, output_dir / "predicted_vs_actual_scatter.png")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(residuals, bins=80, color="#4c78a8", alpha=0.85)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("Residual Distribution", fontweight="bold")
    ax.set_xlabel("Actual - Predicted")
    ax.set_ylabel("Count")
    save_figure(fig, output_dir / "residual_histogram.png")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(residuals[:limit], linewidth=0.8)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Residuals over Test Samples", fontweight="bold")
    ax.set_xlabel("Flattened test sample")
    ax.set_ylabel("Residual")
    ax.grid(True, alpha=0.25)
    save_figure(fig, output_dir / "residual_timeseries.png")


def plot_summary_metrics(targets: np.ndarray, predictions: np.ndarray, output_dir: Path) -> None:
    report = calculate_metrics(targets, predictions).to_dict()
    Path(output_dir, "metrics_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    names = ["mae", "rmse", "mape", "r2", "explained_variance"]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar([name.upper().replace("_", " ") for name in names], [report[name] for name in names], color="#72b7b2")
    ax.set_title("Overall Forecast Metrics", fontweight="bold")
    ax.tick_params(axis="x", rotation=20)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8)
    save_figure(fig, output_dir / "overall_metrics_bar.png")


def plot_model_diagnostics(model: TrafficForecastingModel, output_dir: Path) -> None:
    maps = {name: tensor.detach().cpu().numpy() for name, tensor in model.attention_maps().items()}
    for name, values in maps.items():
        array = values
        while array.ndim > 2:
            array = array.mean(axis=0)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(array, cmap="viridis", ax=ax)
        ax.set_title(f"{name.replace('_', ' ').title()} Heatmap", fontweight="bold")
        save_figure(fig, output_dir / f"{name}_heatmap.png")

    embeddings = model.node_embeddings().detach().cpu().numpy()
    if embeddings.shape[0] >= 3:
        perplexity = max(2, min(30, embeddings.shape[0] // 3))
        coords = TSNE(n_components=2, perplexity=perplexity, init="random", learning_rate="auto").fit_transform(embeddings)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(coords[:, 0], coords[:, 1], s=14, alpha=0.8)
        ax.set_title("Node Embedding t-SNE", fontweight="bold")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.grid(True, alpha=0.2)
        save_figure(fig, output_dir / "node_embedding_tsne.png")


def parse_comparison(values: Optional[Sequence[str]]) -> Dict[str, Dict[str, float]]:
    reports: Dict[str, Dict[str, float]] = {}
    for item in values or []:
        if "=" not in item:
            raise ValueError("--compare-json values must be LABEL=path/to/metrics.json")
        label, path = item.split("=", 1)
        reports[label] = json.loads(Path(path).read_text(encoding="utf-8"))
    return reports


def plot_metric_comparison(reports: Dict[str, Dict[str, float]], output_dir: Path) -> None:
    if not reports:
        return
    metrics = ["mae", "rmse", "mape", "r2", "explained_variance"]
    labels = list(reports)
    fig, axes = plt.subplots(1, len(metrics), figsize=(2.4 * len(metrics), 3.2))
    if len(metrics) == 1:
        axes = [axes]
    for axis, metric in zip(axes, metrics):
        values = [reports[label][metric] for label in labels]
        axis.bar(labels, values, color="#f58518")
        axis.set_title(metric.upper().replace("_", " "), fontweight="bold")
        axis.tick_params(axis="x", rotation=25)
        axis.grid(True, axis="y", alpha=0.25)
    save_figure(fig, output_dir / "dataset_metric_comparison.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate research-grade plots for trained traffic models.")
    parser.add_argument("--config", help="YAML config used for the trained checkpoint.")
    parser.add_argument("--checkpoint", help="Path to best_model.pt checkpoint.")
    parser.add_argument("--output-dir", default="outputs/research_figures", help="Where figures and JSON summaries are written.")
    parser.add_argument("--node-index", type=int, default=0, help="Node/sensor index for horizon line plots.")
    parser.add_argument("--max-points", type=int, default=300, help="Maximum test windows shown in line plots.")
    parser.add_argument("--compare-json", action="append", help="Optional LABEL=path/to/metrics.json for dataset/model comparison bars.")
    parser.add_argument("--dpi", type=int, default=120, help="Saved figure DPI; lower values make smaller PNG files for Colab display.")
    args = parser.parse_args()

    global FIGURE_DPI
    FIGURE_DPI = args.dpi

    output_dir = Path(args.output_dir)
    comparison_reports = parse_comparison(args.compare_json)
    plot_metric_comparison(comparison_reports, output_dir)

    if not args.config or not args.checkpoint:
        if comparison_reports:
            print(f"Saved comparison figures to {output_dir}")
            return
        raise SystemExit("Provide --config and --checkpoint, or at least one --compare-json entry.")

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    predictions, targets, model = collect_predictions(config, args.checkpoint)
    dataset_name = str(config.get("data", {}).get("dataset_name", "Traffic Dataset"))
    frequency_minutes = int(config.get("data", {}).get("frequency_minutes", 5))

    plot_horizon_grid(targets, predictions, output_dir, frequency_minutes, dataset_name, args.node_index, args.max_points)
    plot_per_horizon_metrics(targets, predictions, output_dir, frequency_minutes)
    plot_error_diagnostics(targets, predictions, output_dir)
    plot_summary_metrics(targets, predictions, output_dir)
    plot_model_diagnostics(model, output_dir)
    print(f"Saved research figures to {output_dir}")


if __name__ == "__main__":
    main()