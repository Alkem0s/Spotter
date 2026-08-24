import pandas as pd
import pytest

from src.data.loader import DataLoader


def test_chronological_split_zero_future_leakage():
    dates = pd.date_range("2025-01-01", "2025-10-31", freq="D")
    df = pd.DataFrame({
        "date": dates,
        "load_id": [f"TR-{i:05d}" for i in range(len(dates))],
        "distance": [500.0] * len(dates),
    })

    train_df, val_df, test_df = DataLoader.chronological_split(
        df,
        train_ratio=0.70,
        val_ratio=0.15,
    )

    # Train date maximum must be before or equal to Val date minimum
    assert train_df["date"].max() <= val_df["date"].min()
    # Val date maximum must be before or equal to Test date minimum
    assert val_df["date"].max() <= test_df["date"].min()

    # Total rows must equal sum of split rows
    assert len(train_df) + len(val_df) + len(test_df) == len(df)
