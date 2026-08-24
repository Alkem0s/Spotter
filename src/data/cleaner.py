from __future__ import annotations

import pandas as pd
import numpy as np


class DataCleaner:
    """
    Handles domain-specific anomalies, sign inversions, and missing values
    in freight shipping datasets.
    """

    def __init__(self, median_weight: float = 32000.0):
        self.median_weight = median_weight

    def clean_weights(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Recovers inverted negative weights (e.g. -32000 lbs) via absolute value
        and imputes genuine NaN weights with domain median.
        """
        result = df.copy()
        if "weight" in result.columns:
            result["weight_clean"] = result["weight"].abs().fillna(self.median_weight)
        return result

    def impute_market_index(self, df: pd.DataFrame, reference_df: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        Imputes missing market index values using date-based daily group means
        and linear interpolation to preserve macroeconomic time-series continuity.
        """
        result = df.copy()
        ref = reference_df.copy() if reference_df is not None else result.copy()

        ref["date"] = pd.to_datetime(ref["date"])
        result["date"] = pd.to_datetime(result["date"])

        if "market_index" in ref.columns:
            daily_mi = (
                ref.groupby("date")["market_index"]
                .mean()
                .interpolate(method="time")
                .bfill()
                .ffill()
            )
            if "market_index" in result.columns:
                result["market_index_clean"] = (
                    result["market_index"]
                    .fillna(result["date"].map(daily_mi))
                    .bfill()
                    .ffill()
                    .fillna(1.0)
                )
            else:
                result["market_index_clean"] = result["date"].map(daily_mi).bfill().ffill().fillna(1.0)
        else:
            result["market_index_clean"] = 1.0

        return result

    def clean(self, df: pd.DataFrame, reference_df: pd.DataFrame | None = None) -> pd.DataFrame:
        """Runs full cleaning pipeline on input DataFrame."""
        cleaned = self.clean_weights(df)
        cleaned = self.impute_market_index(cleaned, reference_df=reference_df)
        return cleaned
