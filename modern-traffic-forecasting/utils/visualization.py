from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_history(history: Dict[str, Iterable[float]], output_dir: str) -> None:
    output = Path(output_dir)
    for metric in ["train_loss", "val_loss", "mae", "rmse"]:
        if metric in history and history[metric]:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(list(history[metric]))
            ax.set_title(metric.replace("_", " ").title())
            ax.set_xlabel("Epoch")
            ax.grid(True, alpha=0.3)
            _save(fig, output / f"{metric}.png")


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, output_dir: str, prefix: str = "test") -> None:
    output = Path(output_dir)
    truth = np.asarray(y_true).reshape(-1)
    pred = np.asarray(y_pred).reshape(-1)
    limit = min(500, truth.size)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(truth[:limit], label="Actual")
    ax.plot(pred[:limit], label="Prediction")
    ax.legend()
    ax.set_title("Prediction vs Actual")
    _save(fig, output / f"{prefix}_prediction_vs_actual.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    residuals = truth - pred
    ax.scatter(pred[:limit], residuals[:limit], s=8, alpha=0.6)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Residual Plot")
    _save(fig, output / f"{prefix}_residuals.png")


def plot_attention_heatmaps(attention_maps: Dict[str, np.ndarray], output_dir: str) -> None:
    output = Path(output_dir)
    for name, weights in attention_maps.items():
        arr = np.asarray(weights)
        while arr.ndim > 2:
            arr = arr.mean(axis=0)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(arr, ax=ax, cmap="viridis")
        ax.set_title(f"{name.title()} Attention")
        _save(fig, output / f"attention_{name}.png")


def plot_node_embeddings(embeddings: np.ndarray, output_dir: str) -> None:
    output = Path(output_dir)
    embeddings = np.asarray(embeddings)
    if embeddings.shape[0] < 3:
        return
    perplexity = max(2, min(30, embeddings.shape[0] // 3))
    coords = TSNE(n_components=2, perplexity=perplexity, init="random", learning_rate="auto").fit_transform(embeddings)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(coords[:, 0], coords[:, 1], s=12)
    ax.set_title("Node Embedding t-SNE")
    _save(fig, output / "node_embeddings_tsne.png")