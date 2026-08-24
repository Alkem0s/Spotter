import argparse
import pandas as pd
import numpy as np


def explore_data(filepath="train-test.csv"):
    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"])

    print("DATASET OVERVIEW")
    print(f"Shape: {df.shape}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"Unique pickups: {df['pickup'].nunique()} | Unique deliveries: {df['delivery'].nunique()}")
    print("\nEquipment counts:")
    print(df["equipment"].value_counts())

    print("\nMISSING VALUES AND DATA TYPES")
    print(df.dtypes.to_frame("dtype").join(df.isna().sum().to_frame("null_count")))

    print("\nNUMERIC SUMMARY")
    num_cols = ["distance", "weight", "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "market_index", "quote_signal", "posted_rate"]
    print(df[num_cols].describe().T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]])

    print("\nWEIGHT ANOMALY CHECK")
    print(f"Weights < 0 count: {(df['weight'] < 0).sum()}")
    print(f"Weights == 0 count: {(df['weight'] == 0).sum()}")
    print(f"Weights is NaN count: {df['weight'].isna().sum()}")
    print("Negative weight range:", df.loc[df["weight"] < 0, "weight"].min(), "to", df.loc[df["weight"] < 0, "weight"].max())
    print("Positive weight range:", df.loc[df["weight"] > 0, "weight"].min(), "to", df.loc[df["weight"] > 0, "weight"].max())

    print("\nMARKET INDEX BY DATE")
    daily_mi = df.groupby("date")["market_index"].agg(["count", "mean", "std", "min", "max"])
    print(daily_mi.head(10))
    print(f"Average daily std of market_index: {daily_mi['std'].mean():.4f}")

    print("\nRATE PER MILE (RPM)")
    df["rpm"] = df["posted_rate"] / df["distance"]
    print(df["rpm"].describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]))

    print("\nRPM by equipment:")
    print(df.groupby("equipment")["rpm"].agg(["count", "mean", "median", "std", "min", "max"]))

    print("\nCORRELATIONS WITH TARGET")
    corrs = df[num_cols + ["rpm"]].corr()
    print("Correlation with posted_rate:")
    print(corrs["posted_rate"].sort_values(ascending=False))
    print("\nCorrelation with rpm:")
    print(corrs["rpm"].sort_values(ascending=False))

    print("\nHIGH RPM SUBSET (RPM > 3.50)")
    high_rpm = df[df["rpm"] > 3.5]
    print(f"High RPM rows count: {len(high_rpm)} ({len(high_rpm)/len(df)*100:.2f}%)")
    print(high_rpm[num_cols + ["rpm"]].describe().T[["count", "mean", "std", "min", "50%", "max"]])

    df["is_high_rpm"] = (df["rpm"] > 3.5).astype(int)
    print("\nCorrelation of is_high_rpm with numeric features:")
    print(df[num_cols].apply(lambda s: s.corr(df["is_high_rpm"])).sort_values(ascending=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="train-test.csv")
    args = parser.parse_args()
    explore_data(args.data)
