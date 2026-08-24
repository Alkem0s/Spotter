from __future__ import annotations

from pathlib import Path
from typing import Tuple
import pandas as pd


class DataLoader:
    """
    Loads, sorts, and chronologically splits freight datasets.
    """

    @staticmethod
    def load_dataset(filepath: str | Path) -> pd.DataFrame:
        """Loads CSV dataset and parses datetime."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found at {path}")
        df = pd.read_csv(path)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            if "load_id" in df.columns:
                df = df.sort_values(["date", "load_id"]).reset_index(drop=True)
            else:
                df = df.sort_values("date").reset_index(drop=True)
        return df

    @staticmethod
    def chronological_split(
        df: pd.DataFrame,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits dataset chronologically to prevent temporal data leakage.
        """
        n = len(df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_df = df.iloc[:train_end].copy().reset_index(drop=True)
        val_df = df.iloc[train_end:val_end].copy().reset_index(drop=True)
        test_df = df.iloc[val_end:].copy().reset_index(drop=True)

        return train_df, val_df, test_df
