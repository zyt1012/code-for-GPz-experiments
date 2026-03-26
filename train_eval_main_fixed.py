
from __future__ import annotations

import numpy as np
from config import TrainCfg
from metrics import rmse, nlpd_gaussian

from models_gpflow import fit_full_gpr, fit_svgp, predict_gpflow
from models_gpz_gaussian import fit_gpz_gaussian, predict_gpz_gaussian
from models_gpz_student_t import fit_gpz_student_t, predict_gpz_student_t

# Fixed nu per dataset (as agreed)
NU_MAP = {"concrete": 4.0, "hour": 4.0, "nyc": 2.5}


def _standardize_y(y_tr: np.ndarray):
    y_tr = y_tr.reshape(-1, 1)
    mean = float(y_tr.mean())
    std = float(y_tr.std() + 1e-8)
    return (y_tr - mean) / std, mean, std


def _coverage(y_true: np.ndarray, y_mean: np.ndarray, y_var: np.ndarray, z: float) -> float:
    y = y_true.reshape(-1, 1)
    mu = y_mean.reshape(-1, 1)
    v = np.clip(y_var.reshape(-1, 1), 1e-12, np.inf)
    s = np.sqrt(v)
    lo = mu - z * s
    hi = mu + z * s
    return float(np.mean((y >= lo) & (y <= hi)))


def run_full_gp(X_tr, y_tr, X_te, y_te):
    model = fit_full_gpr(X_tr, y_tr)
    mu, var = predict_gpflow(model, X_te)
    return {
        "rmse": rmse(y_te, mu),
        "nlpd": nlpd_gaussian(y_te, mu, var),
        "cov68": _coverage(y_te, mu, var, z=1.0),
        "cov95": _coverage(y_te, mu, var, z=1.96),
    }


def run_svgp(X_tr, y_tr, X_te, y_te, cfg: TrainCfg):
    model = fit_svgp(
        X_tr, y_tr,
        inducing_m=cfg.inducing_m,
        iters=cfg.iters,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        seed=cfg.seed,
    )
    mu, var = predict_gpflow(model, X_te)
    return {
        "rmse": rmse(y_te, mu),
        "nlpd": nlpd_gaussian(y_te, mu, var),
        "cov68": _coverage(y_te, mu, var, z=1.0),
        "cov95": _coverage(y_te, mu, var, z=1.96),
    }


def run_gpz_gaussian_main(X_tr, y_tr, X_te, y_te, cfg: TrainCfg):
    y_tr_s, mean, std = _standardize_y(y_tr)
    model = fit_gpz_gaussian(
        X_tr, y_tr_s,
        m=cfg.inducing_m,
        iters=cfg.iters,
        lr=cfg.lr,
        seed=cfg.seed,
        batch_size=cfg.batch_size,
        hetero=True,
    )
    mu_s, var_s = predict_gpz_gaussian(model, X_tr, y_tr_s, X_te)
    mu = mu_s * std + mean
    var = var_s * (std ** 2)
    return {
        "rmse": rmse(y_te, mu),
        "nlpd": nlpd_gaussian(y_te, mu, var),
        "cov68": _coverage(y_te, mu, var, z=1.0),
        "cov95": _coverage(y_te, mu, var, z=1.96),
    }


def run_gpz_student_t_main(dataset: str, X_tr, y_tr, X_te, y_te, cfg: TrainCfg, *, debug: bool = False):
    dataset = str(dataset).lower()
    if dataset not in NU_MAP:
        raise ValueError(f"Unknown dataset '{dataset}' for NU_MAP. Expected one of {list(NU_MAP.keys())}.")
    nu = float(NU_MAP[dataset])

    y_tr_s, mean, std = _standardize_y(y_tr)
    model = fit_gpz_student_t(
        X_tr, y_tr_s,
        m=cfg.inducing_m,
        iters=cfg.iters,
        lr=cfg.lr,
        seed=cfg.seed,
        batch_size=cfg.batch_size,
        nu=nu,
        hetero=True,
    )
    mu_s, var_s = predict_gpz_student_t(model, X_tr, y_tr_s, X_te, debug=debug)  # debug default False here

    mu = mu_s * std + mean
    var = var_s * (std ** 2)

    return {
        "rmse": rmse(y_te, mu),
        "nlpd": nlpd_gaussian(y_te, mu, var),
        "cov68": _coverage(y_te, mu, var, z=1.0),
        "cov95": _coverage(y_te, mu, var, z=1.96),
        "nu": nu,
    }
