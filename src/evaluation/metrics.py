from __future__ import annotations

from typing import Dict
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error, median_absolute_error


def evaluate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str = "") -> Dict[str, float]:
    """
    Computes a comprehensive suite of regression metrics including median and tail percentiles.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    abs_errors = np.abs(y_t - y_p)
    pct_errors = abs_errors / np.clip(y_t, 1.0, None) * 100.0

    metrics = {
        f"{prefix}rmse": float(root_mean_squared_error(y_t, y_p)),
        f"{prefix}mae": float(mean_absolute_error(y_t, y_p)),
        f"{prefix}medae": float(median_absolute_error(y_t, y_p)),
        f"{prefix}r2": float(r2_score(y_t, y_p)),
        f"{prefix}mape": float(np.mean(pct_errors)),
        f"{prefix}p90_error": float(np.percentile(abs_errors, 90)),
        f"{prefix}p95_error": float(np.percentile(abs_errors, 95)),
    }
    return metrics


def print_metrics_summary(metrics: Dict[str, float], label: str = "Evaluation Split") -> None:
    """Prints formatted summary of evaluated metrics."""
    print(f"\n==================== {label} ====================")
    print(f"  RMSE (Root Mean Squared Error)   : ${metrics.get('rmse', 0.0):.2f}")
    print(f"  MAE  (Mean Absolute Error)       : ${metrics.get('mae', 0.0):.2f}")
    print(f"  MedAE(Median Absolute Error)     : ${metrics.get('medae', 0.0):.2f}")
    print(f"  R²   (Coefficient of Determ.)    : {metrics.get('r2', 0.0):.4f}")
    print(f"  MAPE (Mean Absolute % Error)     : {metrics.get('mape', 0.0):.2f}%")
    print(f"  P90 Absolute Residual            : ${metrics.get('p90_error', 0.0):.2f}")
    print(f"  P95 Absolute Residual            : ${metrics.get('p95_error', 0.0):.2f}")
    print("==========================================================")
