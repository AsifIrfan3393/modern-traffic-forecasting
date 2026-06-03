import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def generate_extra_research_figures(
    y_true,
    y_pred,
    output_dir="outputs/METR-LA/research_figures_extra",
    model=None
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # =====================================================
    # Error CDF
    # =====================================================

    errors = np.abs(y_true.flatten() - y_pred.flatten())

    sorted_errors = np.sort(errors)

    cdf = np.arange(len(sorted_errors)) / len(sorted_errors)

    plt.figure(figsize=(8, 6))
    plt.plot(sorted_errors, cdf, linewidth=2)

    plt.xlabel("Absolute Error")
    plt.ylabel("CDF")
    plt.title("Error Cumulative Distribution")
    plt.grid(True)

    plt.savefig(
        output_dir / "error_cdf.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # =====================================================
    # Sensor-wise Error Heatmap
    # =====================================================

    if y_true.ndim >= 3:

        sensor_error = np.mean(
            np.abs(y_true - y_pred),
            axis=(0, 1)
        )

        sensor_error = sensor_error.reshape(1, -1)

        plt.figure(figsize=(14, 3))

        sns.heatmap(
            sensor_error,
            cmap="viridis"
        )

        plt.title("Sensor-wise MAE")

        plt.savefig(
            output_dir / "sensor_error_heatmap.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    # =====================================================
    # Correlation Matrix
    # =====================================================

    if y_true.ndim >= 3:

        flattened = y_true.reshape(
            -1,
            y_true.shape[-1]
        )

        corr = np.corrcoef(flattened.T)

        plt.figure(figsize=(10, 8))

        sns.heatmap(
            corr,
            cmap="coolwarm",
            center=0
        )

        plt.title("Traffic Correlation Matrix")

        plt.savefig(
            output_dir / "correlation_matrix.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

    # =====================================================
    # Actual vs Predicted Scatter
    # =====================================================

    plt.figure(figsize=(8, 8))

    plt.scatter(
        y_true.flatten(),
        y_pred.flatten(),
        alpha=0.2
    )

    mn = min(
        y_true.min(),
        y_pred.min()
    )

    mx = max(
        y_true.max(),
        y_pred.max()
    )

    plt.plot(
        [mn, mx],
        [mn, mx],
        "r--"
    )

    plt.xlabel("Actual")
    plt.ylabel("Predicted")

    plt.title("Actual vs Predicted")

    plt.savefig(
        output_dir / "actual_vs_predicted_scatter.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # =====================================================
    # Residual Histogram
    # =====================================================

    residuals = (
        y_true.flatten()
        -
        y_pred.flatten()
    )

    plt.figure(figsize=(8, 6))

    plt.hist(
        residuals,
        bins=50
    )

    plt.xlabel("Residual")

    plt.ylabel("Frequency")

    plt.title("Residual Distribution")

    plt.savefig(
        output_dir / "residual_histogram.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # =====================================================
    # Residual Scatter
    # =====================================================

    plt.figure(figsize=(10, 6))

    plt.scatter(
        y_pred.flatten(),
        residuals,
        alpha=0.2
    )

    plt.axhline(
        0,
        linestyle="--"
    )

    plt.xlabel("Predicted")

    plt.ylabel("Residual")

    plt.title("Residual Plot")

    plt.savefig(
        output_dir / "residual_plot.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # =====================================================
    # Model Complexity
    # =====================================================

    if model is not None:

        total_params = sum(
            p.numel()
            for p in model.parameters()
        )

        trainable_params = sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )

        plt.figure(figsize=(6, 4))

        plt.bar(
            [
                "Total",
                "Trainable"
            ],
            [
                total_params,
                trainable_params
            ]
        )

        plt.title(
            "Model Complexity Analysis"
        )

        plt.ylabel(
            "Number of Parameters"
        )

        plt.savefig(
            output_dir / "model_complexity.png",
            dpi=300,
            bbox_inches="tight"
        )

        plt.close()

        print(
            f"Total Parameters: {total_params:,}"
        )

        print(
            f"Trainable Parameters: {trainable_params:,}"
        )

    print(
        f"Research figures saved to: {output_dir}"
    )