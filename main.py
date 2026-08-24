import os
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, TargetEncoder


def compute_haversine(lon1, lat1, lon2, lat2):
    """Calculates great-circle distance between pickup and delivery in miles."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


def compute_bearing(lat1, lon1, lat2, lon2):
    """Calculates initial heading bearing between coordinates in degrees."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(y, x)) + 360) % 360


def preprocess_data(data, reference_df=None, median_weight=32000.0):
    """
    Cleans and engineers features for freight rate prediction.
    """
    df = data.copy()
    df["date"] = pd.to_datetime(df["date"])

    if "load_id" in df.columns:
        df = df.sort_values(["date", "load_id"]).reset_index(drop=True)
    else:
        df = df.sort_values("date").reset_index(drop=True)

    # Clean corrupted weights
    df["weight_clean"] = df["weight"].abs().fillna(median_weight)

    # Impute missing market index values using daily averages
    if reference_df is not None:
        ref_df = reference_df.copy()
        ref_df["date"] = pd.to_datetime(ref_df["date"])
    else:
        ref_df = df.copy()

    daily_mi = ref_df.groupby("date")["market_index"].mean().interpolate().bfill().ffill()
    df["market_index_clean"] = df["market_index"].fillna(df["date"].map(daily_mi)).bfill().ffill().fillna(1.0)

    # Categorical route representations
    for col in ["pickup", "delivery", "equipment"]:
        df[col] = df[col].astype(str)

    df["origin"] = df["pickup"]
    df["destination"] = df["delivery"]
    df["route"] = df["origin"] + " -> " + df["destination"]
    df["equipment_route"] = df["equipment"] + "_" + df["route"]

    # Spatial geometry and circuity
    df["haversine"] = compute_haversine(
        df["pickup_lon"], df["pickup_lat"], df["delivery_lon"], df["delivery_lat"]
    )
    # Circuity ratio to detect detours
    df["circuity"] = df["distance"] / (df["haversine"] + 1.0)
    df["lat_diff"] = df["delivery_lat"] - df["pickup_lat"]
    df["lon_diff"] = df["delivery_lon"] - df["pickup_lon"]
    df["mid_lat"] = (df["pickup_lat"] + df["delivery_lat"]) / 2.0
    df["mid_lon"] = (df["pickup_lon"] + df["delivery_lon"]) / 2.0
    
    # Heading angles
    df["bearing"] = compute_bearing(
        df["pickup_lat"], df["pickup_lon"], df["delivery_lat"], df["delivery_lon"]
    )
    df["bearing_sin"] = np.sin(np.radians(df["bearing"]))
    df["bearing_cos"] = np.cos(np.radians(df["bearing"]))

    # Physics and freight density features
    df["ton_miles"] = (df["weight_clean"] / 2000.0) * df["distance"]
    df["weight_per_mile"] = df["weight_clean"] / df["distance"].clip(lower=1)
    df["distance_log"] = np.log1p(df["distance"])
    df["weight_log"] = np.log1p(df["weight_clean"])

    # Pricing signals and market interactions
    df["market_x_dist"] = df["market_index_clean"] * df["distance"]
    df["quote_x_dist"] = df["quote_signal"] * df["distance"]
    df["quote_x_market"] = df["quote_signal"] * df["market_index_clean"]
    df["quote_x_market_x_dist"] = df["quote_signal"] * df["market_index_clean"] * df["distance"]
    df["quote_to_market_ratio"] = df["quote_signal"] / (df["market_index_clean"] + 1e-5)

    # Calendar and seasonality
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_year"] = df["date"].dt.dayofyear
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    df["is_quarter_end"] = df["date"].dt.is_quarter_end.astype(int)

    # Cyclical day and month encodings
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)

    # Daily market momentum lags
    daily_stats = (
        ref_df.groupby("date")
        .agg(
            daily_mi=("market_index", "mean"),
            daily_qs=("quote_signal", "mean"),
            daily_count=("load_id", "count") if "load_id" in ref_df.columns else ("pickup", "count"),
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

    df = df.merge(daily_stats.reset_index(), on="date", how="left")
    for lag_col in [
        "daily_mi", "daily_qs", "daily_count", "daily_mi_lag1", "daily_mi_lag7",
        "daily_mi_change7", "daily_qs_lag1", "daily_qs_lag7", "daily_qs_change7", "daily_count_lag1"
    ]:
        df[lag_col] = df[lag_col].bfill().ffill().fillna(0)

    return df


class EnsembleFreightModel:
    """
    Weighted ensemble combining CatBoost (80%), HistGradientBoosting (16%), and Ridge (4%).
    """

    def __init__(self, cat_features, weights=(0.80, 0.16, 0.04), seed=42):
        self.cat_features = cat_features
        self.w_cb, self.w_hgb, self.w_ridge = weights
        self.seed = seed

        self.cb_model = None
        self.hgb_model = None
        self.ridge_model = None
        self.target_encoder = None
        self.scaler = None
        self.feature_cols = None
        self.num_cols = None

    def fit(self, X_train, y_train, X_val=None, y_val=None, iterations=1200):
        self.feature_cols = X_train.columns.tolist()
        self.num_cols = [c for c in self.feature_cols if c not in self.cat_features]

        # CatBoost
        print("Fitting CatBoost...")
        self.cb_model = CatBoostRegressor(
            iterations=iterations,
            learning_rate=0.04,
            depth=6,
            l2_leaf_reg=5.0,
            random_strength=0.5,
            loss_function="RMSE",
            eval_metric="RMSE",
            random_seed=self.seed,
            thread_count=-1,
            verbose=200,
        )
        if X_val is not None and y_val is not None:
            self.cb_model.fit(
                X_train,
                y_train,
                cat_features=self.cat_features,
                eval_set=(X_val, y_val),
                early_stopping_rounds=100,
                verbose=200,
            )
        else:
            self.cb_model.fit(
                X_train,
                y_train,
                cat_features=self.cat_features,
                verbose=200,
            )

        # HistGradientBoosting with K-Fold Target Encoding
        print("Fitting HistGradientBoosting with Target Encoding...")
        cv = KFold(n_splits=5, shuffle=True, random_state=self.seed)
        self.target_encoder = TargetEncoder(smooth="auto", cv=cv)
        tr_cat_enc = pd.DataFrame(
            self.target_encoder.fit_transform(X_train[self.cat_features], y_train),
            columns=[f"{c}_te" for c in self.cat_features],
            index=X_train.index,
        )
        X_tr_hgb = pd.concat([X_train[self.num_cols], tr_cat_enc], axis=1).fillna(0)

        self.hgb_model = HistGradientBoostingRegressor(
            max_iter=400,
            learning_rate=0.05,
            max_depth=8,
            l2_regularization=2.0,
            random_state=self.seed,
        )
        self.hgb_model.fit(X_tr_hgb, y_train)

        # Ridge Regression on standardized features
        print("Fitting Ridge...")
        self.scaler = StandardScaler()
        X_tr_ridge = self.scaler.fit_transform(X_tr_hgb)
        self.ridge_model = RidgeCV(alphas=np.logspace(-2, 4, 20))
        self.ridge_model.fit(X_tr_ridge, y_train)

        print("Ensemble training complete.")

    def predict(self, X_input):
        X = X_input.copy()
        for col in self.cat_features:
            X[col] = X[col].astype(str)

        # CatBoost
        p_cb = self.cb_model.predict(X[self.feature_cols])

        # HistGradientBoosting
        cat_enc = pd.DataFrame(
            self.target_encoder.transform(X[self.cat_features]),
            columns=[f"{c}_te" for c in self.cat_features],
            index=X.index,
        )
        X_hgb = pd.concat([X[self.num_cols], cat_enc], axis=1).fillna(0)
        p_hgb = self.hgb_model.predict(X_hgb)

        # Ridge
        X_ridge = self.scaler.transform(X_hgb)
        p_ridge = self.ridge_model.predict(X_ridge)

        # Weighted blend
        p_blend = self.w_cb * p_cb + self.w_hgb * p_hgb + self.w_ridge * p_ridge
        return np.maximum(p_blend, 10.0)


def print_metrics(y_true, y_pred, split_name="Validation"):
    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"[{split_name}] RMSE: {rmse:.2f} | MAE: {mae:.2f} | R2: {r2:.4f}")
    return {"rmse": rmse, "mae": mae, "r2": r2}


def main():
    # Load data
    train_raw = pd.read_csv("train-test.csv")
    print(f"Loaded {len(train_raw):,} training samples.")

    median_weight = train_raw["weight"].abs().median()
    train_df = preprocess_data(train_raw, median_weight=median_weight)

    cat_features = ["origin", "destination", "pickup", "delivery", "route", "equipment", "equipment_route"]
    ignore_cols = ["load_id", "posted_rate", "date", "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "weight", "market_index"]
    feature_cols = [c for c in train_df.columns if c not in ignore_cols]

    for col in cat_features:
        train_df[col] = train_df[col].astype(str)

    # Chronological Train / Val / Test Split
    # Avoids random split leakage across time periods
    n = len(train_df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    tr = train_df.iloc[:train_end].copy()
    va = train_df.iloc[train_end:val_end].copy()
    te = train_df.iloc[val_end:].copy()

    print(f"\nChronological Split:")
    print(f"  Train : {len(tr):,} samples ({tr['date'].min().strftime('%Y-%m-%d')} to {tr['date'].max().strftime('%Y-%m-%d')})")
    print(f"  Val   : {len(va):,} samples ({va['date'].min().strftime('%Y-%m-%d')} to {va['date'].max().strftime('%Y-%m-%d')})")
    print(f"  Test  : {len(te):,} samples ({te['date'].min().strftime('%Y-%m-%d')} to {te['date'].max().strftime('%Y-%m-%d')})")

    # Model Evaluation on Validation & Test
    print("\nEvaluating Models on Validation Split")
    val_model = EnsembleFreightModel(cat_features=cat_features, weights=(0.80, 0.16, 0.04), seed=42)
    val_model.fit(tr[feature_cols], tr["posted_rate"], va[feature_cols], va["posted_rate"], iterations=1500)

    p_val = val_model.predict(va[feature_cols])
    p_test = val_model.predict(te[feature_cols])

    print("\nValidation & Test Scores")
    print_metrics(va["posted_rate"], p_val, split_name="Val (Aug-Sep)")
    print_metrics(te["posted_rate"], p_test, split_name="Test (Sep-Oct)")

    # Fit Final Model on 100% of train-test data
    print("\nRetraining Final Ensemble on 100% of Data")
    final_model = EnsembleFreightModel(cat_features=cat_features, weights=(0.80, 0.16, 0.04), seed=42)
    final_model.fit(train_df[feature_cols], train_df["posted_rate"], iterations=1000)

    # Generate Predictions for validation.csv
    print("\nGenerating validation_predictions.csv")
    val_raw = pd.read_csv("validation.csv")
    combined_ref = pd.concat([train_raw, val_raw], ignore_index=True)
    val_processed = preprocess_data(val_raw, reference_df=combined_ref, median_weight=median_weight)

    val_preds = final_model.predict(val_processed[feature_cols])

    val_sub = pd.DataFrame({
        "load_id": val_processed["load_id"],
        "predicted_rate": np.round(val_preds, 2),
    })

    # Match template ordering
    template_path = Path("validation-predictions-template.csv")
    if template_path.exists():
        template = pd.read_csv(template_path)
        val_sub = template[["load_id"]].merge(val_sub, on="load_id", how="left")

    val_sub.to_csv("validation_predictions.csv", index=False)
    print(f"Wrote {len(val_sub):,} predictions to validation_predictions.csv")

    # Generate Predictions for december-chart-inputs.csv
    print("\nGenerating december-chart-inputs.csv")
    dec_raw = pd.read_csv("december-chart-inputs.csv")

    # City coordinate mapping from historical data
    city_lat = combined_ref.groupby("pickup")["pickup_lat"].mean().to_dict()
    city_lon = combined_ref.groupby("pickup")["pickup_lon"].mean().to_dict()

    dec_df = dec_raw.copy()
    dec_df["pickup_lat"] = dec_df["pickup"].map(city_lat)
    dec_df["pickup_lon"] = dec_df["pickup"].map(city_lon)
    dec_df["delivery_lat"] = dec_df["delivery"].map(city_lat)
    dec_df["delivery_lon"] = dec_df["delivery"].map(city_lon)

    # Use December daily signals from validation set
    dec_val_rows = val_raw[pd.to_datetime(val_raw["date"]).dt.month == 12]
    dec_daily_mi = dec_val_rows.groupby("date")["market_index"].mean().to_dict()
    dec_daily_qs = dec_val_rows.groupby("date")["quote_signal"].mean().to_dict()

    dec_df["market_index"] = dec_df["date"].map(dec_daily_mi)
    dec_df["quote_signal"] = dec_df["date"].map(dec_daily_qs)

    dec_processed = preprocess_data(dec_df, reference_df=combined_ref, median_weight=median_weight)
    dec_preds = final_model.predict(dec_processed[feature_cols])

    dec_raw["predicted_rate"] = np.round(dec_preds, 2)
    dec_raw.to_csv("december-chart-inputs.csv", index=False)
    print(f"Updated december-chart-inputs.csv with {len(dec_raw)} rows")

    # Scorer validation
    print("\nRunning score.py")
    cmd = [
        "python",
        "score.py",
        "--predictions",
        "validation_predictions.csv",
        "--december-predictions",
        "december-chart-inputs.csv",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("Scorer stderr:", result.stderr)


if __name__ == "__main__":
    main()