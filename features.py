from __future__ import annotations

import numpy as np
import pandas as pd


def _standardize(X: np.ndarray):
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-8
    return (X - mu) / sd


def make_concrete_xy(df: pd.DataFrame):
    """
    Concrete_Data.xls:
    - All numeric columns.
    - Target is assumed to be the last numeric column.
    """
    df = df.copy()
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        raise ValueError("Concrete dataset: expected at least 2 numeric columns (X + y).")

    y = num.iloc[:, -1].to_numpy(dtype=np.float64).reshape(-1, 1)
    X = num.iloc[:, :-1].to_numpy(dtype=np.float64)
    X = _standardize(X)
    return X, y


def make_bike_hour_xy(df: pd.DataFrame):
    """
    hour.csv (UCI Bike Sharing):
    - Predict y = cnt
    - Avoid leakage: DO NOT use casual/registered as inputs.
    - Drop 'dteday' (string date) to keep numeric features only.
    - log1p transform y for stability (common for count-like targets).
    """
    df = df.copy()

    if "cnt" not in df.columns:
        raise ValueError("Bike hour dataset: expected 'cnt' column as target.")

    y = df["cnt"].to_numpy(dtype=np.float64).reshape(-1, 1)

    # Avoid leakage columns + target
    drop_cols = [c for c in ["casual", "registered", "cnt"] if c in df.columns]
    X_df = df.drop(columns=drop_cols, errors="ignore")

    if "dteday" in X_df.columns:
        X_df = X_df.drop(columns=["dteday"])

    # Coerce all to numeric
    X_df = X_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X = X_df.to_numpy(dtype=np.float64)
    X = _standardize(X)

    # Stabilize heavy-tail counts
    y = np.log1p(y)
    return X, y


def make_nyc_xy_from_df(df: pd.DataFrame):
    """
    NYC Taxi Trip Duration (Kaggle train):
    - Target y = trip_duration (seconds), use log1p transform.
    - Minimal, robust feature set:
        vendor_id, passenger_count, pickup/dropoff lat/lon,
        store_and_fwd_flag (0/1),
        pickup hour, weekday, month.
    """
    df = df.copy()

    if "trip_duration" not in df.columns:
        raise ValueError("NYC train dataframe must contain 'trip_duration'.")

    # Datetime parsing
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df["pickup_hour"] = df["pickup_datetime"].dt.hour.astype("float64")
    df["pickup_weekday"] = df["pickup_datetime"].dt.weekday.astype("float64")
    df["pickup_month"] = df["pickup_datetime"].dt.month.astype("float64")

    # store_and_fwd_flag -> 0/1
    if "store_and_fwd_flag" in df.columns:
        df["store_and_fwd_flag"] = (
            df["store_and_fwd_flag"].astype(str).str.upper().eq("Y").astype("float64")
        )
    else:
        df["store_and_fwd_flag"] = 0.0

    feat_cols = [
        "vendor_id", "passenger_count",
        "pickup_longitude", "pickup_latitude",
        "dropoff_longitude", "dropoff_latitude",
        "store_and_fwd_flag",
        "pickup_hour", "pickup_weekday", "pickup_month",
    ]
    for c in feat_cols:
        if c not in df.columns:
            df[c] = 0.0

    X_df = df[feat_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    X = X_df.to_numpy(dtype=np.float64)
    X = _standardize(X)

    y = df["trip_duration"].to_numpy(dtype=np.float64).reshape(-1, 1)
    y = np.log1p(y)
    return X, y
