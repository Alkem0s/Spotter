from __future__ import annotations

import numpy as np
import pandas as pd


def compute_haversine(lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    """Calculates great-circle distance between coordinates in miles."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 3958.8 * 2 * np.arcsin(np.clip(np.sqrt(a), 0.0, 1.0))


def compute_bearing(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Calculates initial heading bearing between coordinates in degrees [0, 360)."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    y = np.sin(dlon) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return (np.degrees(np.arctan2(y, x)) + 360) % 360


def engineer_geospatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers geospatial physics, circuity ratios, directional headings,
    and freight density interaction terms.
    """
    data = df.copy()

    # Route and equipment identifiers
    for col in ["pickup", "delivery", "equipment"]:
        if col in data.columns:
            data[col] = data[col].astype(str)

    data["origin"] = data["pickup"]
    data["destination"] = data["delivery"]
    data["route"] = data["origin"] + " -> " + data["destination"]
    data["equipment_route"] = data["equipment"] + "_" + data["route"]

    # Spatial geometry
    has_coords = {"pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"}.issubset(data.columns)
    if has_coords:
        data["haversine"] = compute_haversine(
            data["pickup_lon"].values,
            data["pickup_lat"].values,
            data["delivery_lon"].values,
            data["delivery_lat"].values,
        )
        data["circuity"] = data["distance"] / (data["haversine"] + 1.0)
        data["lat_diff"] = data["delivery_lat"] - data["pickup_lat"]
        data["lon_diff"] = data["delivery_lon"] - data["pickup_lon"]
        data["mid_lat"] = (data["pickup_lat"] + data["delivery_lat"]) / 2.0
        data["mid_lon"] = (data["pickup_lon"] + data["delivery_lon"]) / 2.0

        data["bearing"] = compute_bearing(
            data["pickup_lat"].values,
            data["pickup_lon"].values,
            data["delivery_lat"].values,
            data["delivery_lon"].values,
        )
        data["bearing_sin"] = np.sin(np.radians(data["bearing"]))
        data["bearing_cos"] = np.cos(np.radians(data["bearing"]))

    # Freight density and physics
    weight_col = "weight_clean" if "weight_clean" in data.columns else "weight"
    if weight_col in data.columns:
        data["ton_miles"] = (data[weight_col] / 2000.0) * data["distance"]
        data["weight_per_mile"] = data[weight_col] / data["distance"].clip(lower=1)
        data["weight_log"] = np.log1p(data[weight_col].clip(lower=0))

    if "distance" in data.columns:
        data["distance_log"] = np.log1p(data["distance"].clip(lower=0))

    # Market interactions
    mi_col = "market_index_clean" if "market_index_clean" in data.columns else "market_index"
    if mi_col in data.columns and "quote_signal" in data.columns:
        data["market_x_dist"] = data[mi_col] * data["distance"]
        data["quote_x_dist"] = data["quote_signal"] * data["distance"]
        data["quote_x_market"] = data["quote_signal"] * data[mi_col]
        data["quote_x_market_x_dist"] = data["quote_signal"] * data[mi_col] * data["distance"]
        data["quote_to_market_ratio"] = data["quote_signal"] / (data[mi_col] + 1e-5)

    return data
