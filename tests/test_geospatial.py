import numpy as np
import pandas as pd
import pytest

from src.features.geospatial import compute_haversine, compute_bearing, engineer_geospatial_features


def test_haversine_known_coordinates():
    # New York City (40.7128, -74.0060) to Los Angeles (34.0522, -118.2437)
    # Expected great-circle distance is approx ~2445 - 2455 miles
    nyc_lat, nyc_lon = 40.7128, -74.0060
    la_lat, la_lon = 34.0522, -118.2437

    dist = compute_haversine(
        np.array([nyc_lon]),
        np.array([nyc_lat]),
        np.array([la_lon]),
        np.array([la_lat]),
    )[0]

    assert 2440.0 < dist < 2460.0


def test_bearing_bounds():
    lat1 = np.array([40.0, 30.0, 50.0])
    lon1 = np.array([-80.0, -90.0, -100.0])
    lat2 = np.array([35.0, 32.0, 45.0])
    lon2 = np.array([-75.0, -85.0, -110.0])

    bearings = compute_bearing(lat1, lon1, lat2, lon2)
    assert (bearings >= 0.0).all()
    assert (bearings < 360.0).all()


def test_engineer_geospatial_features_structure():
    df = pd.DataFrame({
        "pickup": ["Chicago"],
        "delivery": ["Atlanta"],
        "equipment": ["Dry Van"],
        "pickup_lat": [41.8781],
        "pickup_lon": [-87.6298],
        "delivery_lat": [33.7490],
        "delivery_lon": [-84.3880],
        "distance": [715.0],
        "weight_clean": [34000.0],
        "market_index_clean": [1.05],
        "quote_signal": [2.1],
    })

    featured = engineer_geospatial_features(df)

    expected_cols = [
        "route", "equipment_route", "haversine", "circuity",
        "bearing", "bearing_sin", "bearing_cos", "ton_miles",
        "weight_per_mile", "market_x_dist", "quote_x_dist"
    ]
    for col in expected_cols:
        assert col in featured.columns, f"Missing expected column: {col}"
    assert featured.loc[0, "circuity"] > 0
