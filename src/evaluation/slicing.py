from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score


def slice_errors_by_dimension(df: pd.DataFrame, y_true_col: str, y_pred_col: str) -> Dict[str, pd.DataFrame]:
    """
    Slices evaluation errors across business-critical dimensions:
      1. Distance Tiers (Short, Mid, Long haul)
      2. Equipment Classes (Dry Van, Reefer, Flatbed)
      3. Market Volatility Regimes (Standard vs Spot Surges)
    """
    eval_df = df.copy()
    eval_df["abs_error"] = (eval_df[y_true_col] - eval_df[y_pred_col]).abs()
    eval_df["sq_error"] = (eval_df[y_true_col] - eval_df[y_pred_col]) ** 2
    eval_df["rpm_true"] = eval_df[y_true_col] / eval_df["distance"].clip(lower=1)

    # 1. Distance Slices
    def categorize_distance(d: float) -> str:
        if d < 300:
            return "1. Short-Haul (<300 mi)"
        elif d <= 800:
            return "2. Mid-Haul (300-800 mi)"
        else:
            return "3. Long-Haul (>800 mi)"

    eval_df["distance_tier"] = eval_df["distance"].apply(categorize_distance)
    dist_slice = (
        eval_df.groupby("distance_tier")
        .agg(
            count=("abs_error", "count"),
            mean_rate=(y_true_col, "mean"),
            mae=("abs_error", "mean"),
            rmse=("sq_error", lambda x: np.sqrt(x.mean())),
        )
        .reset_index()
    )

    # 2. Equipment Slices
    equip_slice = (
        eval_df.groupby("equipment")
        .agg(
            count=("abs_error", "count"),
            mean_rate=(y_true_col, "mean"),
            mae=("abs_error", "mean"),
            rmse=("sq_error", lambda x: np.sqrt(x.mean())),
        )
        .reset_index()
    )

    # 3. Market Volatility Regime (Surge vs Base)
    eval_df["market_regime"] = np.where(eval_df["rpm_true"] > 3.50, "Spot Surge (RPM > $3.50)", "Standard Market (RPM <= $3.50)")
    regime_slice = (
        eval_df.groupby("market_regime")
        .agg(
            count=("abs_error", "count"),
            pct_dataset=("abs_error", lambda x: len(x) / len(eval_df) * 100),
            mean_rate=(y_true_col, "mean"),
            mae=("abs_error", "mean"),
            rmse=("sq_error", lambda x: np.sqrt(x.mean())),
            total_mse_share=("sq_error", lambda x: x.sum() / eval_df["sq_error"].sum() * 100),
        )
        .reset_index()
    )

    return {
        "distance": dist_slice,
        "equipment": equip_slice,
        "regime": regime_slice,
    }
