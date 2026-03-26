from __future__ import annotations

"""
Report-ready plotting script for the experiment figures currently used in the dissertation.

This script generates the same six figure types already used in the report:
1. RMSE comparison across datasets
2. NLPD comparison across datasets
3. 95% predictive interval coverage across datasets
4. Runtime scaling on Hour and NYC
5. NLPD scaling on Hour and NYC
6. Ablation NLPD comparison across datasets

Usage:
- Run the script.
- Output PNG files will be saved to ./report_figures/
"""

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

LINE_STYLES = {
    "svgp": "-",
    "gpz_gaussian": "--",
    "gpz_student_t": ":",
}

# -----------------------------
# User-adjustable paths
# -----------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR  # change this if your CSV files are elsewhere
OUT_DIR = SCRIPT_DIR / "report_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Global plotting style
# -----------------------------
plt.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.grid": False,
        "lines.linewidth": 1.8,
    }
)

DATASET_ORDER = ["concrete", "hour", "nyc"]
MODEL_ORDER_MAIN = ["fullgp", "svgp", "gpz_gaussian", "gpz_student_t"]
MODEL_ORDER_SCALING = ["svgp", "gpz_gaussian", "gpz_student_t"]
MODEL_ORDER_ABLATION = [
    "svgp",
    "svgp_t",
    "gpz_gaussian",
    "gpz_t_fixed",
    "gpz_t_homo",
    "gpz_t_learnnu",
]

DISPLAY_NAMES: Dict[str, str] = {
    "concrete": "Concrete",
    "hour": "Hour",
    "nyc": "NYC",
    "fullgp": "Full GP",
    "svgp": "SVGP",
    "gpz_gaussian": "GPz-Gaussian",
    "gpz_student_t": "GPz-Student-t",
    "svgp_t": "SVGP-Student-t",
    "gpz_t_fixed": "GPz-Student-t (fixed ν)",
    "gpz_t_homo": "GPz-Student-t (homoscedastic)",
    "gpz_t_learnnu": "GPz-Student-t (learned ν)",
}

BAR_HATCHES = ["", "//", "..", "xx", "\\", "++"]


# -----------------------------
# Helpers
# -----------------------------
def load_csv(filename: str) -> pd.DataFrame:
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path}\n"
            f"Place {filename} next to this script or update DATA_DIR in the script."
        )
    return pd.read_csv(path)


def style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.45)


def save_close(fig: plt.Figure, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def pretty_dataset_names(index_values: List[str]) -> List[str]:
    return [DISPLAY_NAMES.get(v, v) for v in index_values]


def pretty_model_names(columns: List[str]) -> List[str]:
    return [DISPLAY_NAMES.get(v, v) for v in columns]


# -----------------------------
# Figure 1 & 2: Main metric bars
# -----------------------------
def plot_main_metric(df: pd.DataFrame, metric: str, filename: str, title: str, ylabel: str) -> None:
    df = df.copy()
    df = df[df["dataset"].isin(DATASET_ORDER) & df["model"].isin(MODEL_ORDER_MAIN)]
    df["dataset"] = pd.Categorical(df["dataset"], categories=DATASET_ORDER, ordered=True)
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER_MAIN, ordered=True)
    df = df.sort_values(["dataset", "model"])

    pivot = df.pivot(index="dataset", columns="model", values=metric).reindex(DATASET_ORDER)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    pivot.plot(kind="bar", ax=ax, width=0.78, edgecolor="black")

    for i, container in enumerate(ax.containers):
        hatch = BAR_HATCHES[i % len(BAR_HATCHES)]
        for patch in container:
            patch.set_hatch(hatch)
            patch.set_linewidth(0.8)

    ax.set_title(title)
    ax.set_xlabel("Dataset")
    ax.set_ylabel(ylabel)
    ax.set_xticklabels(pretty_dataset_names(DATASET_ORDER), rotation=0)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, pretty_model_names(labels), title="Model", frameon=False, ncols=2)
    style_axes(ax)
    save_close(fig, filename)


# -----------------------------
# Figure 3: Coverage bars
# -----------------------------
def plot_coverage95(df: pd.DataFrame, filename: str) -> None:
    df = df.copy()
    df = df[df["dataset"].isin(DATASET_ORDER) & df["model"].isin(MODEL_ORDER_MAIN)]
    df["dataset"] = pd.Categorical(df["dataset"], categories=DATASET_ORDER, ordered=True)
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER_MAIN, ordered=True)
    df = df.sort_values(["dataset", "model"])

    pivot = df.pivot(index="dataset", columns="model", values="cov95").reindex(DATASET_ORDER)

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    pivot.plot(kind="bar", ax=ax, width=0.78, edgecolor="black")

    for i, container in enumerate(ax.containers):
        hatch = BAR_HATCHES[i % len(BAR_HATCHES)]
        for patch in container:
            patch.set_hatch(hatch)
            patch.set_linewidth(0.8)

    ax.axhline(0.95, linestyle="--", linewidth=1.2, color="black", label="Nominal 95% level")
    ax.set_title("95% Predictive Interval Coverage Across Datasets")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Coverage probability")
    ax.set_xticklabels(pretty_dataset_names(DATASET_ORDER), rotation=0)
    ax.set_ylim(0.0, 1.05)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        pretty_model_names(labels[:-1]) + [labels[-1]],
        title="Legend",
        frameon=False,
        ncols=1,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )
    style_axes(ax)
    save_close(fig, filename)


# -----------------------------
# Figure 4: Runtime scaling
# -----------------------------
def plot_scaling_runtime(df: pd.DataFrame, filename: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2), sharey=False)

    for ax, dataset_key in zip(axes, ["hour", "nyc"]):
        sub = df[(df["dataset"] == dataset_key) & (df["model"].isin(MODEL_ORDER_SCALING))].copy()
        for model in MODEL_ORDER_SCALING:
            m = sub[sub["model"] == model].sort_values("n_train")
            if not m.empty:
                ax.plot(
                    m["n_train"],
                    m["train_seconds"],
                    marker="o",
                    markersize=4.5,
                    linestyle=LINE_STYLES.get(model, "-"),
                    label=DISPLAY_NAMES.get(model, model),
                )
        ax.set_title(f"{DISPLAY_NAMES[dataset_key]}: Runtime scaling")
        ax.set_xlabel("Training set size, n")
        ax.set_ylabel("Training time (seconds)")
        style_axes(ax)
        ax.legend(frameon=False)

    fig.suptitle("Training Time as the Training Set Size Increases", y=1.02, fontsize=12)
    save_close(fig, filename)


# -----------------------------
# Figure 5: NLPD scaling
# -----------------------------
def plot_scaling_nlpd(df: pd.DataFrame, filename: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2), sharey=False)

    for ax, dataset_key in zip(axes, ["hour", "nyc"]):
        sub = df[(df["dataset"] == dataset_key) & (df["model"].isin(MODEL_ORDER_SCALING))].copy()
        for model in MODEL_ORDER_SCALING:
            m = sub[sub["model"] == model].sort_values("n_train")
            if not m.empty:
                ax.plot(
                    m["n_train"],
                    m["nlpd"],
                    marker="o",
                    markersize=4.5,
                    linestyle=LINE_STYLES.get(model, "-"),
                    label=DISPLAY_NAMES.get(model, model),
                )
        ax.set_title(f"{DISPLAY_NAMES[dataset_key]}: NLPD scaling")
        ax.set_xlabel("Training set size, n")
        ax.set_ylabel("Negative log predictive density")
        style_axes(ax)
        ax.legend(frameon=False)

    fig.suptitle("Predictive Density Quality as the Training Set Size Increases", y=1.02, fontsize=12)
    save_close(fig, filename)


# -----------------------------
# Figure 6: Ablation bars
# -----------------------------
def plot_ablation_nlpd(df: pd.DataFrame, filename: str) -> None:
    df = df.copy()
    df = df[df["dataset"].isin(DATASET_ORDER) & df["model"].isin(MODEL_ORDER_ABLATION)]
    df["dataset"] = pd.Categorical(df["dataset"], categories=DATASET_ORDER, ordered=True)
    df["model"] = pd.Categorical(df["model"], categories=MODEL_ORDER_ABLATION, ordered=True)
    df = df.sort_values(["dataset", "model"])

    pivot = df.pivot(index="dataset", columns="model", values="nlpd").reindex(DATASET_ORDER)

    fig, ax = plt.subplots(figsize=(10.2, 4.9))
    pivot.plot(kind="bar", ax=ax, width=0.82, edgecolor="black")

    for i, container in enumerate(ax.containers):
        hatch = BAR_HATCHES[i % len(BAR_HATCHES)]
        for patch in container:
            patch.set_hatch(hatch)
            patch.set_linewidth(0.75)

    ax.set_title("Ablation Study: NLPD Across Model Variants")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Negative log predictive density")
    ax.set_xticklabels(pretty_dataset_names(DATASET_ORDER), rotation=0)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, pretty_model_names(labels), title="Model variant", frameon=False, ncols=2)
    style_axes(ax)
    save_close(fig, filename)


def main() -> None:
    metrics_df = load_csv("results_main_metrics.csv")
    scaling_df = load_csv("results_scaling.csv")
    ablation_df = load_csv("results_ablation.csv")

    plot_main_metric(
        metrics_df,
        metric="rmse",
        filename="figure1_rmse_comparison.png",
        title="RMSE Comparison Across Datasets",
        ylabel="Root mean squared error (RMSE)",
    )

    plot_main_metric(
        metrics_df,
        metric="nlpd",
        filename="figure2_nlpd_comparison.png",
        title="Negative Log Predictive Density Across Datasets",
        ylabel="Negative log predictive density (NLPD)",
    )

    plot_coverage95(metrics_df, filename="figure3_coverage95_comparison.png")
    plot_scaling_runtime(scaling_df, filename="figure4_runtime_scaling.png")
    plot_scaling_nlpd(scaling_df, filename="figure5_scaling_nlpd.png")
    plot_ablation_nlpd(ablation_df, filename="figure6_ablation_nlpd.png")

    print(f"Saved report-ready figures to: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
