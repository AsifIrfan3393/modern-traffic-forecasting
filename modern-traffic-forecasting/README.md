# Modern Spatio-Temporal Traffic Forecasting

This repository refactors the original ISTGCN-style notebook into a modular, research-grade traffic forecasting framework.  The new system keeps the same problem statement—traffic flow/speed prediction from road-sensor time series—but modernizes the architecture with adaptive graph learning, GATv2, multi-scale temporal encoding, Temporal Fusion Transformer components, spatial/temporal attention, and production training utilities.

## Architecture

```text
Input traffic features
  → cleaning, missing-value masks, scaling, calendar features
  → adaptive/static/hybrid graph learning
  → stacked GATv2 spatial encoder
  → multi-scale temporal branches for 15/30/60 minute patterns
  → Temporal Fusion Transformer encoder/decoder
  → spatial attention + temporal attention
  → feature fusion with trainable node embeddings
  → dense prediction head
  → multi-horizon traffic forecast
```

## Project Layout

```text
configs/config.yaml                         Experiment configuration
dataset/traffic_dataset.py                  Dataset loading, windows, graph inputs
models/adaptive_graph.py                    Static/adaptive/hybrid adjacency learning
models/node_embeddings.py                   Trainable node embeddings
models/gatv2_encoder.py                     Residual LayerNorm GATv2 blocks
models/temporal_fusion_transformer.py       VSN, GRN, transformer attention
models/multiscale_encoder.py                15/30/60-minute temporal branches
models/spatial_attention.py                 Node-level attention
models/temporal_attention.py                Time-step attention
models/feature_fusion.py                    Attention-based feature fusion
models/predictor.py                         End-to-end forecasting model
trainers/trainer.py                         AdamW, AMP, warmup cosine, early stopping
utils/preprocessing.py                      Cleaning, scaling, time features
utils/metrics.py                            MAE, RMSE, MAPE, R², explained variance
utils/visualization.py                      Curves, forecasts, residuals, heatmaps, t-SNE
train.py                                    Training entry point
evaluate.py                                 Test-set evaluation entry point
inference.py                                Single-window forecasting entry point
```

## Data Format

The loader accepts common traffic arrays:

- `[time, nodes]`
- `[time, nodes, features]`
- `[nodes, time, features]` for legacy PeMS-style arrays

Supported file types include `.npy`, `.npz`, `.csv`, `.txt`, `.h5`, and `.hdf5`.  Optional adjacency and edge-feature files can be configured in `configs/config.yaml`.

## Quick Start

```bash
pip install -r requirements.txt
python train.py --config configs/config.yaml
python evaluate.py --config configs/config.yaml --checkpoint outputs/checkpoints/best_model.pt
python inference.py --config configs/config.yaml --checkpoint outputs/checkpoints/best_model.pt
```

Update `configs/config.yaml` so `data.data_path`, `data.adjacency_path`, and optional `data.edge_features_path` point to METR-LA, PEMS-BAY, PeMSD4, PeMSD8, or another compatible dataset.

## Key Features

- **Adaptive graph learning:** `A = softmax(E @ E.T)` with static, adaptive, and hybrid modes.
- **Hybrid graph fusion:** trainable `α` combines fixed roads with learned dependencies.
- **GATv2 spatial modeling:** multi-head attention, residual paths, dropout, LayerNorm, and edge attributes when available.
- **Temporal Fusion Transformer:** variable selection, gated residual networks, transformer attention, and learnable temporal positional embeddings.
- **Multi-scale temporal branches:** configurable branch kernels defaulting to 15, 30, and 60 minute patterns for 5-minute traffic data.
- **Feature engineering:** missing masks, normalized numerical variables, time-of-day, day-of-week, month, weekend, and holiday indicators.
- **Training stack:** AdamW, weight decay, warmup + cosine schedule, early stopping, gradient clipping, and CUDA AMP.
- **Evaluation:** MAE, RMSE, MAPE, R², and explained variance with JSON reports.
- **Visualization:** loss/metric curves, prediction-vs-actual plots, residuals, attention heatmaps, and t-SNE node embeddings.

## Configuration

`configs/config.yaml` exposes model, graph, optimizer, scheduler, loss, batch size, horizon, AMP, and reproducibility settings.  Use `model.graph_mode` to switch among `static`, `adaptive`, and `hybrid` graph learning.