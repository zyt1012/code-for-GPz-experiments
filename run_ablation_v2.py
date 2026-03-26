from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict

import numpy as np

from config import Paths, TrainCfg
from datasets import load_concrete, load_bike_hour, load_nyc_taxi
from metrics import rmse, nlpd_gaussian
from models_gpflow import fit_full_gpr, fit_svgp, predict_gpflow
from models_gpflow_student_t import fit_svgp_student_t, predict_svgp_student_t
from models_gpz_student_t import fit_gpz_student_t, predict_gpz_student_t
from models_gpz_gaussian import fit_gpz_gaussian, predict_gpz_gaussian
from models_gpz_student_t_learnnu_irls import fit_gpz_student_t_learnnu, predict_gpz_student_t_learnnu

NU_MAP = {"concrete": 4.0, "hour": 4.0, "nyc": 2.5}


def coverage(y_true: np.ndarray, mu: np.ndarray, var: np.ndarray, z: float) -> float:
    y_true = y_true.reshape(-1, 1)
    mu = mu.reshape(-1, 1)
    var = var.reshape(-1, 1)
    std = np.sqrt(np.maximum(var, 1e-12))
    lo = mu - z * std
    hi = mu + z * std
    return float(np.mean((y_true >= lo) & (y_true <= hi)))


def standardize_y(y_tr: np.ndarray):
    y_tr = y_tr.reshape(-1, 1)
    y_mean = float(y_tr.mean())
    y_std = float(y_tr.std() + 1e-8)
    y_tr_s = (y_tr - y_mean) / y_std
    return y_tr_s, y_mean, y_std


def load_dataset(dataset: str, paths: Paths, cfg: TrainCfg):
    if dataset == "concrete":
        return load_concrete(paths, cfg)
    if dataset == "hour":
        return load_bike_hour(paths, cfg)
    if dataset == "nyc":
        return load_nyc_taxi(paths, cfg)[:4]
    raise ValueError(dataset)


def run_one(dataset: str, model_name: str, paths: Paths, cfg: TrainCfg, debug: bool = False) -> Dict:
    loaded = load_dataset(dataset, paths, cfg)
    X_tr, X_te, y_tr, y_te = loaded
    y_tr = y_tr.reshape(-1, 1)
    y_te = y_te.reshape(-1, 1)

    t0 = time.perf_counter()

    if model_name == "fullgp":
        if dataset == "nyc" or X_tr.shape[0] > 3000:
            raise RuntimeError("Full GP not run for this dataset/size.")
        model = fit_full_gpr(X_tr, y_tr)
        mu, var = predict_gpflow(model, X_te)
        meta = {"nu": "", "hetero": "", "arch": "kernel", "lik": "Gaussian"}
    elif model_name == "svgp":
        model = fit_svgp(X_tr, y_tr, inducing_m=cfg.inducing_m, iters=cfg.iters, batch_size=cfg.batch_size, lr=cfg.lr, seed=cfg.seed)
        mu, var = predict_gpflow(model, X_te)
        meta = {"nu": "", "hetero": "homo", "arch": "kernel", "lik": "Gaussian"}
    elif model_name == "svgp_t":
        nu = NU_MAP[dataset]
        model = fit_svgp_student_t(X_tr, y_tr, inducing_m=cfg.inducing_m, iters=cfg.iters, batch_size=cfg.batch_size, lr=cfg.lr, seed=cfg.seed, nu=nu)
        mu, var = predict_svgp_student_t(model, X_te)
        meta = {"nu": nu, "hetero": "homo", "arch": "kernel", "lik": "StudentT"}
    elif model_name == "gpz_gaussian":
        y_tr_s, y_mean, y_std = standardize_y(y_tr)
        model = fit_gpz_gaussian(X_tr, y_tr_s, m=cfg.inducing_m, iters=cfg.iters, lr=cfg.lr, seed=cfg.seed, batch_size=cfg.batch_size, hetero=True)
        mu_s, var_s = predict_gpz_gaussian(model, X_tr, y_tr_s, X_te, debug=False)
        mu = mu_s * y_std + y_mean
        var = var_s * (y_std ** 2)
        meta = {"nu": "", "hetero": "hetero", "arch": "basis", "lik": "Gaussian"}
    elif model_name == "gpz_t_fixed":
        nu = NU_MAP[dataset]
        y_tr_s, y_mean, y_std = standardize_y(y_tr)
        model = fit_gpz_student_t(X_tr, y_tr_s, m=cfg.inducing_m, iters=cfg.iters, lr=cfg.lr, seed=cfg.seed, batch_size=cfg.batch_size, nu=nu, hetero=True)
        mu_s, var_s = predict_gpz_student_t(model, X_tr, y_tr_s, X_te, debug=debug)
        mu = mu_s * y_std + y_mean
        var = var_s * (y_std ** 2)
        meta = {"nu": nu, "hetero": "hetero", "arch": "basis", "lik": "StudentT"}
    elif model_name == "gpz_t_learnnu":
        nu_init = NU_MAP[dataset]
        y_tr_s, y_mean, y_std = standardize_y(y_tr)
        model = fit_gpz_student_t_learnnu(X_tr, y_tr_s, m=cfg.inducing_m, iters=cfg.iters, lr=cfg.lr, seed=cfg.seed, batch_size=cfg.batch_size, nu_init=nu_init, hetero=True)
        mu_s, var_s = predict_gpz_student_t_learnnu(model, X_tr, y_tr_s, X_te, debug=debug)
        mu = mu_s * y_std + y_mean
        var = var_s * (y_std ** 2)
        meta = {"nu": float(model.nu().numpy()), "hetero": "hetero", "arch": "basis", "lik": "StudentT"}
    elif model_name == "gpz_t_homo":
        nu = NU_MAP[dataset]
        y_tr_s, y_mean, y_std = standardize_y(y_tr)
        model = fit_gpz_student_t(X_tr, y_tr_s, m=cfg.inducing_m, iters=cfg.iters, lr=cfg.lr, seed=cfg.seed, batch_size=cfg.batch_size, nu=nu, hetero=False)
        mu_s, var_s = predict_gpz_student_t(model, X_tr, y_tr_s, X_te, debug=debug)
        mu = mu_s * y_std + y_mean
        var = var_s * (y_std ** 2)
        meta = {"nu": nu, "hetero": "homo", "arch": "basis", "lik": "StudentT"}
    else:
        raise ValueError(model_name)

    t1 = time.perf_counter()

    row = {
        "dataset": dataset,
        "model": model_name,
        "n_train": int(X_tr.shape[0]),
        "n_test": int(X_te.shape[0]),
        "rmse": float(rmse(y_te, mu)),
        "nlpd": float(nlpd_gaussian(y_te, mu, var)),
        "cov68": float(coverage(y_te, mu, var, z=1.0)),
        "cov95": float(coverage(y_te, mu, var, z=1.96)),
        "train_seconds": float(t1 - t0),
        "inducing_m": int(cfg.inducing_m),
        "iters": int(cfg.iters),
        "batch_size": int(cfg.batch_size),
        "lr": float(cfg.lr),
        "seed": int(cfg.seed),
        **meta,
    }
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_csv", default="results_ablation.csv")
    ap.add_argument("--datasets", default="concrete,hour,nyc")
    ap.add_argument("--models", default="svgp,svgp_t,gpz_gaussian,gpz_t_fixed,gpz_t_homo,gpz_t_learnnu")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    paths = Paths()
    cfg = TrainCfg()

    datasets = [s.strip() for s in args.datasets.split(",") if s.strip()]
    models = [s.strip() for s in args.models.split(",") if s.strip()]

    rows = []
    for ds in datasets:
        for m in models:
            try:
                row = run_one(ds, m, paths, cfg, debug=args.debug)
                rows.append(row)
                print(f"[ABLATION][{ds}|{m}] RMSE={row['rmse']:.4f} NLPD={row['nlpd']:.4f} cov95={row['cov95']:.4f} time={row['train_seconds']:.2f}s nu={row.get('nu','')}")
            except Exception as e:
                print(f"[SKIP][{ds}|{m}] {e}")

    out = Path(args.out_csv)
    if rows:
        fieldnames = list(rows[0].keys())
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\n[OK] wrote {len(rows)} rows -> {out.resolve()}")
    else:
        print("No rows produced.")

if __name__ == "__main__":
    main()
