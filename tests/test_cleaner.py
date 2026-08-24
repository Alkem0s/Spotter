import numpy as np
import pandas as pd
import pytest

from src.data.cleaner import DataCleaner


def test_clean_weights_negative_inversion():
    cleaner = DataCleaner(median_weight=30000.0)
    df = pd.DataFrame({
        "weight": [-32000.0, -15000.0, 45000.0, np.nan],
    })
    cleaned = cleaner.clean_weights(df)

    assert "weight_clean" in cleaned.columns
    assert (cleaned["weight_clean"] > 0).all()
    assert cleaned.loc[0, "weight_clean"] == 32000.0
    assert cleaned.loc[1, "weight_clean"] == 15000.0
    assert cleaned.loc[2, "weight_clean"] == 45000.0
    assert cleaned.loc[3, "weight_clean"] == 30000.0


def test_impute_market_index_interpolation():
    cleaner = DataCleaner()
    df = pd.DataFrame({
        "date": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"],
        "market_index": [1.0, np.nan, 1.2, np.nan],
    })
    cleaned = cleaner.impute_market_index(df)

    assert "market_index_clean" in cleaned.columns
    assert not cleaned["market_index_clean"].isna().any()
    assert (cleaned["market_index_clean"] > 0).all()
    assert np.isclose(cleaned.loc[1, "market_index_clean"], 1.1)  # Linear interpolation between 1.0 and 1.2
