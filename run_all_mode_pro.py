
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from config import Paths, TrainCfg
from datasets import load_concrete, load_bike_hour, load_nyc_taxi
from train_eval_main_fixed import (
    run_full_gp,
    run_svgp,
    run_gpz_gaussian_main,
    run_gpz_student_t_main,
)

# Mainline model sets (per your write-up requirement)
# concrete/hour: 4 models (fullgp, svgp, gpz_gaussian, gpz_student_t)
# nyc: 3 models (svgp, gpz_gaussian, gpz_student_t)  (fullgp not feasible)
MAINLINE_MODELS = {
    "concrete": ["fullgp", "svgp", "gpz_gaussian", "gpz_student_t"],
    "hour": ["fullgp", "svgp", "gpz_gaussian", "gpz_student_t"],
    "nyc": ["svgp", "gpz_gaussian", "gpz_student_t"],
}

# Scaling models: focus on scalable methods (SVGP + GPz variants)
SCALING_MODELS = ["svgp", "gpz_gaussian", "gpz_student_t"]


def _time_call(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    t1 = time.perf_counter()
    return out, (t1 - t0)


def _subsample(X: np.ndarray, y: np.ndarray, n: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    n = int(min(n, X.shape[0]))
    if n == X.shape[0]:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(X.shape[0], size=n, replace=False)
    return X[idx], y[idx]


def _print_metrics(res: Dict):
    print(f"RMSE = {res['rmse']:.4f}")
    print(f"NLPD = {res['nlpd']:.4f}")
    print(f"Coverage 68% = {res['cov68']:.4f} | Coverage 95% = {res['cov95']:.4f}")
    if "nu" in res:
        print(f"nu = {float(res['nu'])}")


def _load_nyc(paths: Paths, cfg: TrainCfg):
    out = load_nyc_taxi(paths, cfg)
    # Support both NYC loader signatures (either 4 or 5 returns)
    if isinstance(out, (list, tuple)) and len(out) == 5:
        X_tr, X_te, y_tr, y_te, _ = out
        return X_tr, X_te, y_tr, y_te
    return out  # type: ignore


def run_metrics(paths: Paths, cfg: TrainCfg, debug_gpz: bool) -> List[Dict]:
    rows: List[Dict] = []

    # ----- Concrete -----
    X_tr, X_te, y_tr, y_te = load_concrete(paths, cfg)
    for model in MAINLINE_MODELS["concrete"]:
        print(f"\n========== Running concrete | {model} ==========")
        if model == "fullgp":
            res, secs = _time_call(run_full_gp, X_tr, y_tr, X_te, y_te)
        elif model == "svgp":
            res, secs = _time_call(run_svgp, X_tr, y_tr, X_te, y_te, cfg)
        elif model == "gpz_gaussian":
            res, secs = _time_call(run_gpz_gaussian_main, X_tr, y_tr, X_te, y_te, cfg)
        else:
            res, secs = _time_call(run_gpz_student_t_main, "concrete", X_tr, y_tr, X_te, y_te, cfg, debug=debug_gpz)

        _print_metrics(res)
        rows.append({
            "mode": "metrics",
            "dataset": "concrete",
            "model": model,
            "n_train": int(X_tr.shape[0]),
            "n_test": int(X_te.shape[0]),
            "train_seconds": float(secs),
            "rmse": float(res["rmse"]),
            "nlpd": float(res["nlpd"]),
            "cov68": float(res["cov68"]),
            "cov95": float(res["cov95"]),
            "nu": float(res["nu"]) if "nu" in res else "",
        })

    # ----- Hour -----
    X_tr, X_te, y_tr, y_te = load_bike_hour(paths, cfg)
    for model in MAINLINE_MODELS["hour"]:
        X_use, y_use = X_tr, y_tr
        # Full GP subsample for hour
        if model == "fullgp":
            max_n = getattr(cfg, "hour_fullgp_max_train", 3000)
            if max_n and X_tr.shape[0] > max_n:
                X_use, y_use = _subsample(X_tr, y_tr, int(max_n), seed=cfg.seed)
                print(f"[hour|fullgp] subsample train: {int(max_n)}/{X_tr.shape[0]}")

        print(f"\n========== Running hour | {model} ==========")
        if model == "fullgp":
            res, secs = _time_call(run_full_gp, X_use, y_use, X_te, y_te)
        elif model == "svgp":
            res, secs = _time_call(run_svgp, X_use, y_use, X_te, y_te, cfg)
        elif model == "gpz_gaussian":
            res, secs = _time_call(run_gpz_gaussian_main, X_use, y_use, X_te, y_te, cfg)
        else:
            res, secs = _time_call(run_gpz_student_t_main, "hour", X_use, y_use, X_te, y_te, cfg, debug=debug_gpz)

        _print_metrics(res)
        rows.append({
            "mode": "metrics",
            "dataset": "hour",
            "model": model,
            "n_train": int(X_use.shape[0]),
            "n_test": int(X_te.shape[0]),
            "train_seconds": float(secs),
            "rmse": float(res["rmse"]),
            "nlpd": float(res["nlpd"]),
            "cov68": float(res["cov68"]),
            "cov95": float(res["cov95"]),
            "nu": float(res["nu"]) if "nu" in res else "",
        })

    # ----- NYC -----
    X_tr, X_te, y_tr, y_te = _load_nyc(paths, cfg)
    for model in MAINLINE_MODELS["nyc"]:
        print(f"\n========== Running nyc | {model} ==========")
        if model == "svgp":
            res, secs = _time_call(run_svgp, X_tr, y_tr, X_te, y_te, cfg)
        elif model == "gpz_gaussian":
            res, secs = _time_call(run_gpz_gaussian_main, X_tr, y_tr, X_te, y_te, cfg)
        else:
            res, secs = _time_call(run_gpz_student_t_main, "nyc", X_tr, y_tr, X_te, y_te, cfg, debug=debug_gpz)

        _print_metrics(res)
        rows.append({
            "mode": "metrics",
            "dataset": "nyc",
            "model": model,
            "n_train": int(X_tr.shape[0]),
            "n_test": int(X_te.shape[0]),
            "train_seconds": float(secs),
            "rmse": float(res["rmse"]),
            "nlpd": float(res["nlpd"]),
            "cov68": float(res["cov68"]),
            "cov95": float(res["cov95"]),
            "nu": float(res["nu"]) if "nu" in res else "",
        })

    return rows


def run_scaling(paths: Paths, cfg: TrainCfg, hour_grid: List[int], nyc_grid: List[int], debug_gpz: bool) -> List[Dict]:
    rows: List[Dict] = []

    # -------- Hour scaling --------
    X_tr, X_te, y_tr, y_te = load_bike_hour(paths, cfg)
    for n in hour_grid:
        Xn, yn = _subsample(X_tr, y_tr, n=n, seed=cfg.seed)
        print(f"\n========== Scaling hour | n={int(Xn.shape[0])} ==========")
        for model in SCALING_MODELS:
            t0 = time.time()
            if model == "svgp":
                res, secs = _time_call(run_svgp, Xn, yn, X_te, y_te, cfg)
                nu_val = ""
            elif model == "gpz_gaussian":
                res, secs = _time_call(run_gpz_gaussian_main, Xn, yn, X_te, y_te, cfg)
                nu_val = ""
            else:
                res, secs = _time_call(run_gpz_student_t_main, "hour", Xn, yn, X_te, y_te, cfg, debug=debug_gpz)
                nu_val = float(res.get("nu", "")) if "nu" in res else ""

            print(f"[scaling] hour | {model:<12} n={int(Xn.shape[0])} time={secs:.2f}s rmse={res['rmse']:.4f} nlpd={res['nlpd']:.4f}")
            rows.append({
                "mode": "scaling",
                "dataset": "hour",
                "model": model,
                "n_train": int(Xn.shape[0]),
                "n_test": int(X_te.shape[0]),
                "train_seconds": float(secs),
                "rmse": float(res["rmse"]),
                "nlpd": float(res["nlpd"]),
                "cov68": float(res["cov68"]),
                "cov95": float(res["cov95"]),
                "nu": nu_val,
            })

    # -------- NYC scaling --------
    X_tr, X_te, y_tr, y_te = _load_nyc(paths, cfg)
    for n in nyc_grid:
        Xn, yn = _subsample(X_tr, y_tr, n=n, seed=cfg.seed)
        print(f"\n========== Scaling nyc | n={int(Xn.shape[0])} ==========")
        for model in SCALING_MODELS:
            if model == "svgp":
                res, secs = _time_call(run_svgp, Xn, yn, X_te, y_te, cfg)
                nu_val = ""
            elif model == "gpz_gaussian":
                res, secs = _time_call(run_gpz_gaussian_main, Xn, yn, X_te, y_te, cfg)
                nu_val = ""
            else:
                res, secs = _time_call(run_gpz_student_t_main, "nyc", Xn, yn, X_te, y_te, cfg, debug=debug_gpz)
                nu_val = float(res.get("nu", "")) if "nu" in res else ""

            print(f"[scaling] nyc  | {model:<12} n={int(Xn.shape[0])} time={secs:.2f}s rmse={res['rmse']:.4f} nlpd={res['nlpd']:.4f}")
            rows.append({
                "mode": "scaling",
                "dataset": "nyc",
                "model": model,
                "n_train": int(Xn.shape[0]),
                "n_test": int(X_te.shape[0]),
                "train_seconds": float(secs),
                "rmse": float(res["rmse"]),
                "nlpd": float(res["nlpd"]),
                "cov68": float(res["cov68"]),
                "cov95": float(res["cov95"]),
                "nu": nu_val,
            })

    return rows


def main():
    ap = argparse.ArgumentParser(description="Final paper runner: metrics + scaling in one script.")
    ap.add_argument("--mode", choices=["metrics", "scaling", "both"], default="metrics")
    ap.add_argument("--out_csv", default="results_main_all.csv")
    ap.add_argument("--hour_grid", default="5000,10000,20000,50000", help="Comma-separated train sizes for Hour scaling.")
    ap.add_argument("--nyc_grid", default="20000,50000,100000,200000", help="Comma-separated train sizes for NYC scaling.")
    ap.add_argument("--debug_gpz", action="store_true", help="Print GPz debug stats (default off).")
    args = ap.parse_args()

    paths = Paths()
    cfg = TrainCfg()

    rows: List[Dict] = []
    if args.mode in ("metrics", "both"):
        rows.extend(run_metrics(paths, cfg, debug_gpz=args.debug_gpz))

    if args.mode in ("scaling", "both"):
        hour_grid = [int(x.strip()) for x in args.hour_grid.split(",") if x.strip()]
        nyc_grid = [int(x.strip()) for x in args.nyc_grid.split(",") if x.strip()]
        rows.extend(run_scaling(paths, cfg, hour_grid=hour_grid, nyc_grid=nyc_grid, debug_gpz=args.debug_gpz))

    df = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"\n[OK] Saved {args.out_csv} ({len(df)} rows).")


if __name__ == "__main__":
    main()
