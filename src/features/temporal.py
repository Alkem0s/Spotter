from __future__ import annotations

import numpy as np
import pandas as pd


def engineer_temporal_features(df: pd.DataFrame, reference_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Engineers calendar indicators, cyclical trigonometric date encodings,
    and causal, leak-free daily market momentum indicators.
    """
    data = df.copy()
    data["date"] = pd.to_datetime(data["date"])

    # Calendar features
    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month
    data["day_of_week"] = data["date"].dt.dayofweek
    data["day_of_year"] = data["date"].dt.dayofyear
    data["is_weekend"] = (data["day_of_week"] >= 5).astype(int)
    data["is_month_end"] = data["date"].dt.is_month_end.astype(int)
    data["is_quarter_end"] = data["date"].dt.is_quarter_end.astype(int)

    # Cyclical representations
    data["dow_sin"] = np.sin(2 * np.pi * data["day_of_week"] / 7.0)
    data["dow_cos"] = np.cos(2 * np.pi * data["day_of_week"] / 7.0)
    data["month_sin"] = np.sin(2 * np.pi * (data["month"] - 1) / 12.0)
    data["month_cos"] = np.cos(2 * np.pi * (data["month"] - 1) / 12.0)

    # Daily market momentum & lag features
    ref = reference_df.copy() if reference_df is not None else data.copy()
    ref["date"] = pd.to_datetime(ref["date"])

    mi_col = "market_index_clean" if "market_index_clean" in ref.columns else "market_index"
    has_signals = {mi_col, "quote_signal"}.issubset(ref.columns)

    if has_signals:
        daily_stats = (
            ref.groupby("date")
            .agg(
                daily_mi=(mi_col, "mean"),
                daily_qs=("quote_signal", "mean"),
                daily_count=("load_id", "count") if "load_id" in ref.columns else ("pickup", "count"),
            )
            .sort_index()
        )
        daily_stats["daily_mi"] = daily_stats["daily_mi"].interpolate().bfill().ffill()
        daily_stats["daily_mi_lag1"] = daily_stats["daily_mi"].shift(1)
        daily_stats["daily_mi_lag7"] = daily_stats["daily_mi"].shift(7)
        daily_stats["daily_mi_change7"] = daily_stats["daily_mi"] - daily_stats["daily_mi_lag7"]
        daily_stats["daily_qs_lag1"] = daily_stats["daily_qs"].shift(1)
        daily_stats["daily_qs_lag7"] = daily_stats["daily_qs"].shift(7)
        daily_stats["daily_qs_change7"] = daily_stats["daily_qs"] - daily_stats["daily_qs_lag7"]
        daily_stats["daily_count_lag1"] = daily_stats["daily_count"].shift(1)

        data = data.merge(daily_stats.reset_index(), on="date", how="left")
        lag_cols = [
            "daily_mi", "daily_qs", "daily_count", "daily_mi_lag1", "daily_mi_lag7",
            "daily_mi_change7", "daily_qs_lag1", "daily_qs_lag7", "daily_qs_change7", "daily_count_lag1"
        ]
        for col in lag_cols:
            if col in data.columns:
                data[col] = data[col].bfill().ffill().fillna(0)

    return data
