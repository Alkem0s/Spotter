from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from src.data.loader import DataLoader
from src.data.cleaner import DataCleaner
from src.features.geospatial import engineer_geospatial_features
from src.features.temporal import engineer_temporal_features
from src.models.ensemble import EnsembleFreightModel
from src.evaluation.metrics import evaluate_regression_metrics, print_metrics_summary
from src.evaluation.slicing import slice_errors_by_dimension


def load_config(config_path: str = "configs/default_config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_pipeline(config_path: str = "configs/default_config.yaml") -> None:
    cfg = load_config(config_path)
    print("==================================================================")
    print("      FREIGHT RATE PRICING INTELLIGENCE & FORECASTING ENGINE      ")
    print("==================================================================")

    # 1. Load Data
    print(f"\n[1/6] Loading datasets from {cfg['data']['train_path']}...")
    train_raw = DataLoader.load_dataset(cfg['data']['train_path'])
    print(f"      Loaded {len(train_raw):,} labeled loads ({train_raw['date'].min().date()} to {train_raw['date'].max().date()}).")

    # 2. Data Cleaning & Feature Engineering
    print("\n[2/6] Running Data Cleaning & Feature Engineering...")
    cleaner = DataCleaner(median_weight=cfg['preprocessing']['default_median_weight'])
    train_clean = cleaner.clean(train_raw)
    train_geo = engineer_geospatial_features(train_clean)
    train_featured = engineer_temporal_features(train_geo, reference_df=train_clean)

    cat_features = cfg['features']['categorical']
    ignore_cols = cfg['features']['ignore']
    feature_cols = [c for c in train_featured.columns if c not in ignore_cols]

    for col in cat_features:
        train_featured[col] = train_featured[col].astype(str)

    print(f"      Constructed {len(feature_cols)} active features (Geospatial, Routing, Temporal, Interaction).")

    # 3. Chronological Out-of-Time Splitting
    print("\n[3/6] Splitting Dataset Chronologically...")
    tr_df, va_df, te_df = DataLoader.chronological_split(
        train_featured,
        train_ratio=cfg['split']['train_ratio'],
        val_ratio=cfg['split']['val_ratio'],
    )
    print(f"      Train Set : {len(tr_df):,} loads ({tr_df['date'].min().date()} to {tr_df['date'].max().date()})")
    print(f"      Val Set   : {len(va_df):,} loads ({va_df['date'].min().date()} to {va_df['date'].max().date()})")
    print(f"      Test Set  : {len(te_df):,} loads ({te_df['date'].min().date()} to {te_df['date'].max().date()})")

    # 4. Out-of-Time Model Evaluation
    print("\n[4/6] Training & Evaluating Out-of-Time Ensemble Model...")
    eval_model = EnsembleFreightModel(
        cat_features=cat_features,
        weights=(
            cfg['model']['weights']['catboost'],
            cfg['model']['weights']['hist_gb'],
            cfg['model']['weights']['ridge'],
        ),
        target_mode=cfg['model']['target_mode'],
        floor_rate=cfg['preprocessing']['floor_prediction_rate'],
        seed=cfg['model']['seed'],
    )
    eval_model.fit(
        tr_df[feature_cols],
        tr_df["posted_rate"],
        va_df[feature_cols],
        va_df["posted_rate"],
        distances_train=tr_df["distance"],
        distances_val=va_df["distance"],
        cb_iterations=cfg['model']['catboost']['iterations'],
        cb_learning_rate=cfg['model']['catboost']['learning_rate'],
        cb_depth=cfg['model']['catboost']['depth'],
        early_stopping_rounds=cfg['model']['catboost']['early_stopping_rounds'],
    )

    va_preds = eval_model.predict(va_df[feature_cols], distances=va_df["distance"])
    te_preds = eval_model.predict(te_df[feature_cols], distances=te_df["distance"])

    val_metrics = evaluate_regression_metrics(va_df["posted_rate"].values, va_preds)
    test_metrics = evaluate_regression_metrics(te_df["posted_rate"].values, te_preds)

    print_metrics_summary(val_metrics, label="Validation Split Metrics (Out-of-Time)")
    print_metrics_summary(test_metrics, label="Test Split Metrics (Out-of-Time)")

    # Error Slicing
    va_eval = va_df.copy()
    va_eval["y_pred"] = va_preds
    slices = slice_errors_by_dimension(va_eval, "posted_rate", "y_pred")
    print("\n--- Error Slicing by Distance Tier ---")
    print(slices["distance"].to_string(index=False))
    print("\n--- Error Slicing by Equipment Class ---")
    print(slices["equipment"].to_string(index=False))
    print("\n--- Error Slicing by Market Volatility Regime ---")
    print(slices["regime"].to_string(index=False))

    # Save figures
    fig_dir = Path(cfg['data']['figures_dir'])
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Feature Importance Plot
    importances = eval_model.get_feature_importances().head(15)
    plt.figure(figsize=(10, 6), dpi=150)
    sns.barplot(x=importances.values, y=importances.index, hue=importances.index, palette="viridis", legend=False)
    plt.title("Top 15 Feature Importances (CatBoost)", fontsize=14, fontweight="bold")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(fig_dir / "feature_importance.png")
    plt.close()

    # Residual Distribution Plot
    residuals = va_df["posted_rate"].values - va_preds
    plt.figure(figsize=(9, 5), dpi=150)
    sns.histplot(residuals[np.abs(residuals) < 500], bins=50, kde=True, color="#064A56")
    plt.title("Validation Residual Distribution (Standard Market Range)", fontsize=14, fontweight="bold")
    plt.xlabel("Residual Error ($: Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(fig_dir / "residual_distribution.png")
    plt.close()

    # 5. Full Dataset Retraining & Inference
    print("\n[5/6] Retraining Production Ensemble on 100% of Development Data...")
    prod_model = EnsembleFreightModel(
        cat_features=cat_features,
        weights=(
            cfg['model']['weights']['catboost'],
            cfg['model']['weights']['hist_gb'],
            cfg['model']['weights']['ridge'],
        ),
        target_mode=cfg['model']['target_mode'],
        floor_rate=cfg['preprocessing']['floor_prediction_rate'],
        seed=cfg['model']['seed'],
    )
    prod_model.fit(
        train_featured[feature_cols],
        train_featured["posted_rate"],
        distances_train=train_featured["distance"],
        cb_iterations=1000,
        cb_learning_rate=0.04,
        cb_depth=6,
    )

    # Generate validation predictions
    print(f"\n[6/6] Generating Output Predictions...")
    val_raw = DataLoader.load_dataset(cfg['data']['val_path'])
    combined_ref = pd.concat([train_raw, val_raw], ignore_index=True)
    
    val_clean = cleaner.clean(val_raw, reference_df=combined_ref)
    val_geo = engineer_geospatial_features(val_clean)
    val_featured = engineer_temporal_features(val_geo, reference_df=combined_ref)
    for col in cat_features:
        val_featured[col] = val_featured[col].astype(str)

    val_preds = prod_model.predict(val_featured[feature_cols], distances=val_featured["distance"])
    val_sub = pd.DataFrame({
        "load_id": val_featured["load_id"],
        "predicted_rate": np.round(val_preds, 2),
    })

    template_path = Path(cfg['data']['template_path'])
    if template_path.exists():
        template = pd.read_csv(template_path)
        val_sub = template[["load_id"]].merge(val_sub, on="load_id", how="left")

    val_out_path = Path(cfg['data']['output_predictions_path'])
    val_sub.to_csv(val_out_path, index=False)
    print(f"      Saved {len(val_sub):,} predictions to {val_out_path}")

    # Generate December Benchmark Predictions
    dec_raw = DataLoader.load_dataset(cfg['data']['december_path'])
    city_lat = combined_ref.groupby("pickup")["pickup_lat"].mean().to_dict()
    city_lon = combined_ref.groupby("pickup")["pickup_lon"].mean().to_dict()

    dec_df = dec_raw.copy()
    dec_df["pickup_lat"] = dec_df["pickup"].map(city_lat)
    dec_df["pickup_lon"] = dec_df["pickup"].map(city_lon)
    dec_df["delivery_lat"] = dec_df["delivery"].map(city_lat)
    dec_df["delivery_lon"] = dec_df["delivery"].map(city_lon)

    dec_val_rows = val_raw[pd.to_datetime(val_raw["date"]).dt.month == 12]
    dec_daily_mi = dec_val_rows.groupby("date")["market_index"].mean().to_dict()
    dec_daily_qs = dec_val_rows.groupby("date")["quote_signal"].mean().to_dict()

    dec_df["market_index"] = dec_df["date"].map(dec_daily_mi)
    dec_df["quote_signal"] = dec_df["date"].map(dec_daily_qs)

    dec_clean = cleaner.clean(dec_df, reference_df=combined_ref)
    dec_geo = engineer_geospatial_features(dec_clean)
    dec_featured = engineer_temporal_features(dec_geo, reference_df=combined_ref)
    for col in cat_features:
        dec_featured[col] = dec_featured[col].astype(str)

    dec_preds = prod_model.predict(dec_featured[feature_cols], distances=dec_featured["distance"])
    dec_raw["predicted_rate"] = np.round(dec_preds, 2)
    dec_out_path = Path(cfg['data']['output_december_path'])
    dec_raw.to_csv(dec_out_path, index=False)
    print(f"      Updated {dec_out_path} with {len(dec_raw)} daily benchmark forecasts.")

    # Benchmark verification
    print("\n--- Verifying Benchmark Outputs ---")
    cmd = [
        "python",
        "scripts/validate_benchmark.py",
        "--predictions",
        str(val_out_path),
        "--december-predictions",
        str(dec_out_path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print("Benchmark validator error:", res.stderr)

    print("\n[SUCCESS] Pipeline execution finished cleanly.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Freight Rate Pricing Pipeline")
    parser.add_argument("--config", default="configs/default_config.yaml", help="Path to YAML configuration")
    args = parser.parse_args()
    run_pipeline(args.config)
