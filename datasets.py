from __future__ import annotations

import pandas as pd
from sklearn.model_selection import train_test_split

from config import Paths, TrainCfg
from features import make_concrete_xy, make_bike_hour_xy, make_nyc_xy_from_df


def load_concrete(paths: Paths, cfg: TrainCfg):
    df = pd.read_excel(paths.concrete_xls)
    X, y = make_concrete_xy(df)
    return train_test_split(X, y, test_size=cfg.test_size, random_state=cfg.seed)


def load_bike_hour(paths: Paths, cfg: TrainCfg):
    df = pd.read_csv(paths.bike_hour_csv)
    X, y = make_bike_hour_xy(df)
    return train_test_split(X, y, test_size=cfg.test_size, random_state=cfg.seed)


def load_nyc_taxi(paths: Paths, cfg: TrainCfg):
    # Read a subset from the huge NYC train file
    df = pd.read_csv(
        paths.nyc_train_csv,
        usecols=list(cfg.nyc_usecols),
        nrows=cfg.nyc_max_rows,
    )
    X, y = make_nyc_xy_from_df(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=cfg.test_size, random_state=cfg.seed)

    # Kaggle test has no labels; loaded only for optional future "predict-only"
    try:
        df_test = pd.read_csv(paths.nyc_test_csv, nrows=200_000)
    except Exception:
        df_test = None

    return X_tr, X_te, y_tr, y_te, df_test
