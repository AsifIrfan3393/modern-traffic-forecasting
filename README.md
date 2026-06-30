# 🚦 Modern Traffic Forecasting using Adaptive Graph Learning and Temporal Fusion Networks

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Research%20Project-success)

A state-of-the-art deep learning framework for **multi-horizon traffic forecasting** that combines **Adaptive Graph Learning**, **Graph Attention Networks (GATv2)**, **Multi-Scale Temporal Encoding**, and **Temporal Fusion Transformers (TFT)** to accurately model complex spatial and temporal dependencies in road traffic networks.

---

# 📌 Overview

Traffic forecasting is a fundamental problem in Intelligent Transportation Systems (ITS). Traditional approaches struggle to model long-range temporal patterns and dynamic spatial relationships among traffic sensors.

This project proposes a hybrid architecture that learns:

- Dynamic road network connectivity
- Spatial dependencies using Graph Neural Networks
- Long-term temporal dependencies
- Multi-horizon traffic prediction

The model is designed to achieve high forecasting accuracy while remaining scalable for real-world traffic networks.

---

# ✨ Features

- Adaptive Graph Learning
- Graph Attention Network (GATv2)
- Multi-Scale Temporal Encoding
- Temporal Fusion Transformer
- Multi-Horizon Forecasting
- Mixed Precision Training (AMP)
- AdamW Optimizer
- Cosine Warmup Scheduler
- Huber Loss
- Early Stopping
- Gradient Clipping
- Comprehensive Evaluation Metrics
- Visualization Utilities

---

# 🏗 Model Architecture

```
Historical Traffic Data
            │
            ▼
 Data Preprocessing & Normalization
            │
            ▼
 Adaptive Graph Learning
            │
            ▼
   Graph Attention Network (GATv2)
            │
            ▼
 Multi-Scale Temporal Encoder
            │
            ▼
 Temporal Fusion Transformer
            │
            ▼
 Multi-Horizon Traffic Prediction
```

---

# 📂 Project Structure

```
modern-traffic-forecasting/
│
├── configs/                  # Configuration files
├── data/                     # Datasets
├── datasets/                 # Dataset loaders
├── models/                   # Deep learning models
├── outputs/
│   ├── checkpoints/
│   ├── figures/
│   └── metrics/
├── scripts/
├── utils/
├── train.py                  # Training pipeline
├── evaluate.py               # Model evaluation
├── inference.py              # Prediction script
├── requirements.txt
└── README.md
```

---

# 🧠 Model Components

## 1. Adaptive Graph Learning

Instead of relying on a fixed adjacency matrix, the model dynamically learns relationships among traffic sensors.

Benefits:

- Dynamic spatial dependency learning
- Better adaptability
- Improved robustness

---

## 2. Graph Attention Network (GATv2)

Captures spatial interactions between neighboring sensors using attention mechanisms.

Advantages:

- Learns importance of neighboring nodes
- Handles heterogeneous traffic conditions
- Better feature aggregation

---

## 3. Multi-Scale Temporal Encoder

Extracts temporal features across different time scales.

Captures:

- Short-term traffic fluctuations
- Daily periodicity
- Weekly seasonal patterns
- Long-range dependencies

---

## 4. Temporal Fusion Transformer (TFT)

Performs multi-horizon forecasting using attention mechanisms.

Benefits:

- Long sequence modeling
- Temporal attention
- Improved forecasting accuracy
- Better interpretability

---

# 📊 Dataset

The framework supports benchmark traffic datasets including:

- METR-LA
- PEMS-BAY
- PEMSD4
- PEMSD8

Input features include:

- Traffic speed
- Historical observations
- Temporal information
- Missing-value masks
- Calendar features

---

# ⚙ Preprocessing

The preprocessing pipeline includes:

- Missing value handling
- Z-score normalization
- Sliding window generation
- Calendar feature engineering
- Train/Validation/Test split

---

# 🚀 Training

Training utilizes several optimization techniques:

- AdamW Optimizer
- Cosine Warmup Learning Rate Scheduler
- Huber Loss
- Automatic Mixed Precision (AMP)
- Gradient Clipping
- Early Stopping

---

# 📈 Evaluation Metrics

The model is evaluated using:

- MAE
- RMSE
- MAPE
- R² Score
- Explained Variance
- SMAPE
- WAPE
- Pearson Correlation
- NSE

Additional visualizations include:

- Prediction vs Ground Truth
- Residual Distribution
- Error Heatmaps
- Sensor-wise Performance
- Error CDF
- Learning Curves

---

# 💻 Installation

Clone the repository

```bash
git clone https://github.com/AsifIrfan3393/modern-traffic-forecasting.git
cd modern-traffic-forecasting
```

Create virtual environment

```bash
python -m venv venv
```

Activate environment

Windows

```bash
venv\Scripts\activate
```

Linux

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶ Training

```bash
python train.py
```

---

# 📊 Evaluation

```bash
python evaluate.py
```

---

# 🔮 Inference

```bash
python inference.py
```

---

# 📉 Sample Results

| Metric | Score |
|---------|-------|
| MAE | 5.25 |
| RMSE | 12.59 |
| MAPE | 11.99% |
| R² Score | 0.695 |
| Explained Variance | 0.698 |

---

# 🔍 Key Contributions

- Hybrid Graph Neural Network + Transformer architecture
- Dynamic graph learning for evolving traffic networks
- Multi-scale temporal representation
- Robust training strategy
- Comprehensive evaluation framework
- Extensible modular implementation

---

# 🛣 Applications

- Smart Cities
- Intelligent Transportation Systems
- Traffic Management
- Route Planning
- Congestion Prediction
- Urban Mobility Analytics
- Autonomous Driving

---

# 🛠 Future Improvements

- Weather-aware forecasting
- Accident-aware prediction
- Explainable AI (XAI)
- Real-time deployment
- Edge computing support
- Federated learning
- Large-scale citywide forecasting

---

# 📚 Tech Stack

- Python
- PyTorch
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- NetworkX

---

# 🤝 Contributing

Contributions are welcome!

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to your branch
5. Open a Pull Request

---

# 📜 License

This project is released under the **MIT License**.

---

# 👨‍💻 Author

**Asif Irfan**

B.Tech Information Technology

Passionate about

- Deep Learning
- Graph Neural Networks
- Computer Vision
- Machine Learning
- Intelligent Transportation Systems

If you found this project useful, consider giving it a ⭐ on GitHub!
