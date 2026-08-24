from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPECTED_ROWS = 12_000
EXPECTED_IDS = {f"TE-{index:06d}" for index in range(1, EXPECTED_ROWS + 1)}
DECEMBER_DATES = pd.date_range("2025-12-01", "2025-12-31", freq="D")
FIXED_PICKUP = "Lexington"
FIXED_DELIVERY = "Fort Wayne"
FIXED_DISTANCE = 360.0
FIXED_EQUIPMENT = "Dry Van"
FIXED_WEIGHT = 32_000.0


def fail(message: str) -> None:
    raise SystemExit(f"ERROR: {message}")


def read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.is_file():
        fail(f"{label} file not found: {path}")
    try:
        return pd.read_csv(path)
    except Exception as exc:
        fail(f"could not read {label}: {exc}")


def numeric_series(frame: pd.DataFrame, column: str, label: str) -> pd.Series:
    values = pd.to_numeric(frame[column], errors="coerce")
    if values.isna().any() or not np.isfinite(values).all():
        fail(f"{label} contains invalid {column} values")
    return values.astype(float)


def validate_predictions(predictions: pd.DataFrame) -> None:
    if list(predictions.columns) != ["load_id", "predicted_rate"]:
        fail("predictions must contain exactly two columns in this order: load_id,predicted_rate")
    if len(predictions) != EXPECTED_ROWS:
        fail(f"predictions must contain exactly {EXPECTED_ROWS:,} rows")
    if predictions["load_id"].isna().any() or predictions["load_id"].duplicated().any():
        fail("predictions contains missing or duplicate load_id values")

    submitted_ids = set(predictions["load_id"].astype(str))
    missing = EXPECTED_IDS - submitted_ids
    extra = submitted_ids - EXPECTED_IDS
    if missing or extra:
        fail(
            "prediction IDs do not match the validation set "
            f"(missing={len(missing)}, extra={len(extra)})"
        )

    predicted_rate = numeric_series(predictions, "predicted_rate", "predictions")
    if (predicted_rate <= 0).any():
        fail("predictions contains non-positive predicted_rate values")


def validate_december(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["pickup", "delivery", "distance", "equipment", "weight", "date", "predicted_rate"]
    if list(frame.columns) != columns:
        fail("December predictions must keep the original seven columns and column order")

    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    if result["date"].isna().any():
        fail("December predictions contains invalid dates")
    result["distance"] = numeric_series(result, "distance", "December predictions")
    result["weight"] = numeric_series(result, "weight", "December predictions")
    result["predicted_rate"] = numeric_series(result, "predicted_rate", "December predictions")

    if result["date"].duplicated().any():
        fail("December predictions contains duplicate dates")
    if len(result) != 31 or set(result["date"]) != set(DECEMBER_DATES):
        fail("December predictions must contain one row for every day from 2025-12-01 to 2025-12-31")
    if not result["pickup"].eq(FIXED_PICKUP).all():
        fail(f"December pickup must be {FIXED_PICKUP} for all rows")
    if not result["delivery"].eq(FIXED_DELIVERY).all():
        fail(f"December delivery must be {FIXED_DELIVERY} for all rows")
    if not np.isclose(result["distance"], FIXED_DISTANCE).all():
        fail(f"December distance must be {FIXED_DISTANCE:g} for all rows")
    if not result["equipment"].eq(FIXED_EQUIPMENT).all():
        fail(f"December equipment must be {FIXED_EQUIPMENT} for all rows")
    if not np.isclose(result["weight"], FIXED_WEIGHT).all():
        fail(f"December weight must be {FIXED_WEIGHT:g} for all rows")
    if (result["predicted_rate"] <= 0).any():
        fail("December predicted_rate values must be positive")
    return result.sort_values("date")


def save_december_chart(december: pd.DataFrame, output: Path) -> None:
    figure, axis = plt.subplots(figsize=(11.5, 5.0), dpi=200)
    color = "#0A5C6B"
    axis.plot(
        december["date"],
        december["predicted_rate"],
        color=color,
        linewidth=2.8,
        marker="o",
        markersize=4.0,
        label="Forecasted Spot Rate ($)",
    )
    floor = float(december["predicted_rate"].min())
    axis.fill_between(
        december["date"],
        december["predicted_rate"],
        floor - max(10.0, floor * 0.02),
        color=color,
        alpha=0.12,
    )

    # Highlight Holiday / Month End events
    axis.axvline(pd.to_datetime("2025-12-25"), color="#C0392B", linestyle="--", alpha=0.6, linewidth=1.2)
    axis.text(
        pd.to_datetime("2025-12-25"),
        floor + 15,
        " Christmas",
        color="#C0392B",
        fontsize=9,
        fontweight="bold",
        rotation=90,
    )

    axis.axvline(pd.to_datetime("2025-12-31"), color="#27AE60", linestyle="--", alpha=0.6, linewidth=1.2)
    axis.text(
        pd.to_datetime("2025-12-31"),
        floor + 15,
        " Year-End Surge",
        color="#27AE60",
        fontsize=9,
        fontweight="bold",
        rotation=90,
    )

    axis.set_title("Benchmark Corridor: December 2025 Dynamic Rate Forecast", loc="left", fontsize=14, fontweight="bold", pad=12)
    axis.set_ylabel("Predicted Spot Rate ($)", fontsize=11, fontweight="bold")
    axis.grid(axis="y", color="#D9E2E4", linewidth=0.8, linestyle=":")
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#9DAFB3")
    axis.tick_params(axis="x", rotation=35)
    axis.text(
        0,
        -0.35,
        "Benchmark Parameters: Lexington, KY to Fort Wayne, IN | 360 miles | Dry Van | 32,000 lbs Payload",
        transform=axis.transAxes,
        fontsize=9.5,
        color="#455A60",
    )
    axis.legend(loc="upper left", frameon=True)
    figure.tight_layout(rect=(0, 0.10, 1, 1))
    figure.savefig(output, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate output prediction schemas and generate benchmark corridor forecasts."
    )
    parser.add_argument("--predictions", required=True, help="CSV containing [load_id, predicted_rate]")
    parser.add_argument(
        "--december-predictions",
        required=True,
        help="CSV containing 31 daily benchmark forecasts",
    )
    parser.add_argument("--output-dir", default="reports/figures")
    args = parser.parse_args()

    validate_predictions(read_csv(Path(args.predictions), "predictions"))
    december = validate_december(read_csv(Path(args.december_predictions), "December predictions"))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    
    chart_primary = output / "benchmark_december_forecast.png"
    save_december_chart(december, chart_primary)

    print(f"Validated {EXPECTED_ROWS:,} out-of-time predictions.")
    print("Validated 31 daily benchmark corridor forecasts.")
    print(f"Generated benchmark visualization: {chart_primary}")


if __name__ == "__main__":
    main()
