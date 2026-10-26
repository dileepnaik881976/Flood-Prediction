"""Matplotlib chart builders shared by evaluate.py (PNG export) and the Streamlit dashboard.

Figures are built with matplotlib.figure.Figure (no pyplot global state), so they are safe to
create repeatedly inside Streamlit reruns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from . import config

BLUE = "#1F6FB2"
ORANGE = "#E07B39"
GREY = "#6B7280"


def plot_target_distribution(y: pd.Series) -> Figure:
    fig = Figure(figsize=(8, 4.2), layout="constrained")
    ax = fig.subplots()
    ax.hist(y, bins=60, color=BLUE, edgecolor="white")
    ax.axvline(y.mean(), color=ORANGE, linestyle="--", label=f"Mean = {y.mean():.4f}")
    ax.set_xlabel("FloodProbability (dataset target)")
    ax.set_ylabel("Number of rows")
    ax.set_title("Distribution of FloodProbability")
    ax.legend()
    return fig


def plot_feature_distributions(df: pd.DataFrame) -> Figure:
    fig = Figure(figsize=(14, 10), layout="constrained")
    axes = fig.subplots(5, 4).ravel()
    for ax, col in zip(axes, config.FEATURES):
        counts = df[col].value_counts().sort_index()
        ax.bar(counts.index, counts.values, color=BLUE)
        ax.set_title(col, fontsize=8)
        ax.tick_params(labelsize=7)
    fig.supxlabel("Feature value (integer score)")
    fig.supylabel("Number of rows")
    fig.suptitle("Distribution of each input feature (random sample)")
    return fig


def plot_correlation_heatmap(corr: pd.DataFrame) -> Figure:
    off_diag = corr.to_numpy(copy=True)
    np.fill_diagonal(off_diag, 0.0)
    limit = max(float(np.abs(off_diag).max()), 0.05)
    fig = Figure(figsize=(9, 8), layout="constrained")
    ax = fig.subplots()
    image = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-limit, vmax=limit)
    ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr.index)), corr.index, fontsize=7)
    fig.colorbar(image, ax=ax, label="Pearson correlation")
    ax.set_title(f"Correlation heatmap (colour scale +/-{limit:.2f}, diagonal ignored)\n"
                 "Correlation does not prove causation")
    return fig


def plot_metric_comparison(models: dict) -> Figure:
    names = list(models)
    fig = Figure(figsize=(12, 4.2), layout="constrained")
    axes = fig.subplots(1, 3)
    for ax, (metric, label) in zip(axes, [("MAE", "MAE (lower is better)"),
                                          ("RMSE", "RMSE (lower is better)"),
                                          ("R2", "R² (higher is better)")]):
        values = [models[n][metric] for n in names]
        ax.bar(range(len(names)), values, color=BLUE)
        ax.set_xticks(range(len(names)), names, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(label)
        ax.set_title(metric)
    fig.suptitle("Validation metrics by model")
    return fig


def plot_actual_vs_predicted(actual, predicted, max_points: int = 5000, seed: int = 42) -> Figure:
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(actual), size=min(max_points, len(actual)), replace=False)
    lo, hi = float(min(actual.min(), predicted.min())), float(max(actual.max(), predicted.max()))
    fig = Figure(figsize=(6.5, 6), layout="constrained")
    ax = fig.subplots()
    ax.scatter(actual[idx], predicted[idx], s=6, alpha=0.3, color=BLUE)
    ax.plot([lo, hi], [lo, hi], color=ORANGE, linestyle="--", label="Perfect prediction")
    ax.set_xlabel("Actual FloodProbability (validation)")
    ax.set_ylabel("Predicted FloodProbability")
    ax.set_title(f"Actual vs predicted ({len(idx):,} random validation rows)")
    ax.legend()
    return fig


def plot_residuals(actual, predicted, max_points: int = 5000, seed: int = 42) -> Figure:
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    residual = actual - predicted
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(actual), size=min(max_points, len(actual)), replace=False)
    fig = Figure(figsize=(12, 4.5), layout="constrained")
    ax1, ax2 = fig.subplots(1, 2)
    ax1.hist(residual, bins=60, color=BLUE, edgecolor="white")
    ax1.axvline(0, color=ORANGE, linestyle="--")
    ax1.set_xlabel("Residual (actual - predicted)")
    ax1.set_ylabel("Number of validation rows")
    ax1.set_title(f"Residual distribution (mean {residual.mean():.4f}, std {residual.std():.4f})")
    ax2.scatter(predicted[idx], residual[idx], s=6, alpha=0.3, color=BLUE)
    ax2.axhline(0, color=ORANGE, linestyle="--")
    ax2.set_xlabel("Predicted FloodProbability")
    ax2.set_ylabel("Residual (actual - predicted)")
    ax2.set_title("Residuals vs predicted value")
    return fig


def plot_feature_importance(importance: pd.DataFrame) -> Figure:
    data = importance.sort_values("importance_mean")
    fig = Figure(figsize=(8, 6.5), layout="constrained")
    ax = fig.subplots()
    ax.barh(data["feature"], data["importance_mean"], xerr=data["importance_std"], color=BLUE)
    ax.set_xlabel("Increase in validation RMSE when the feature is shuffled")
    ax.set_title("Permutation feature importance (model-agnostic)")
    ax.tick_params(axis="y", labelsize=8)
    return fig


def plot_feature_contributions(contributions: pd.Series) -> Figure:
    data = contributions.sort_values()
    fig = Figure(figsize=(7, 6), layout="constrained")
    ax = fig.subplots()
    ax.barh(data.index, data.values, color=[ORANGE if v < 0 else BLUE for v in data.values])
    ax.axvline(0, color=GREY, linewidth=0.8)
    ax.set_xlabel("Change vs dataset-average input (percentage points)")
    # Display only: the plotted data stay in score units; tick labels show them x 100.
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _pos: f"{x * 100:+.1f}"))
    ax.set_title("How each input moves this prediction")
    ax.tick_params(axis="y", labelsize=8)
    return fig
