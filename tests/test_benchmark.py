from pathlib import Path
import pandas as pd
import pytest

from scripts.validate_benchmark import validate_predictions, validate_december, read_csv


def test_validation_predictions_schema():
    pred_path = Path("validation_predictions.csv")
    if not pred_path.exists():
        pytest.skip("validation_predictions.csv not yet generated")
    
    df = read_csv(pred_path, "predictions")
    validate_predictions(df)
    assert len(df) == 12_000
    assert (df["predicted_rate"] > 0).all()


def test_december_benchmark_schema():
    dec_path = Path("data/december_benchmark.csv")
    if not dec_path.exists():
        dec_path = Path("december-chart-inputs.csv")
    if not dec_path.exists():
        pytest.skip("benchmark file not found")

    df = read_csv(dec_path, "december")
    validated = validate_december(df)
    assert len(validated) == 31
    assert (validated["predicted_rate"] > 0).all()
    assert (validated["distance"] == 360.0).all()
    assert (validated["equipment"] == "Dry Van").all()
