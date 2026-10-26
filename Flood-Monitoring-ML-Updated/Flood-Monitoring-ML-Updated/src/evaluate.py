"""Build evaluation artifacts WITHOUT retraining: figures, data summary, dashboard sample.

Run from the project root:  python -m src.evaluate
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from . import charts, config, data_processing as dp
from .train import regression_metrics

FIGURES_DIR = config.REPORTS_DIR / "figures"
DATA_SUMMARY_PATH = config.REPORTS_DIR / "data_summary.json"
TRAIN_SAMPLE_PATH = config.REPORTS_DIR / "train_sample.csv"
SAMPLE_ROWS = 50_000


def _ranges(df: pd.DataFrame) -> dict:
    return {c: [int(df[c].min()), int(df[c].max())] for c in config.FEATURES}


def build_data_summary(train: pd.DataFrame, test: pd.DataFrame, flood: pd.DataFrame) -> dict:
    cols = config.FEATURES + [config.TARGET]
    desc = train[config.FEATURES].describe(percentiles=[0.25, 0.5, 0.75]).T
    corr = train[cols].corr()
    return {
        "train": {"rows": int(len(train)), "columns": int(train.shape[1]),
                  "missing_values": int(train.isna().sum().sum()),
                  "duplicate_feature_rows": int(train.duplicated(subset=config.FEATURES).sum()),
                  "feature_ranges": _ranges(train)},
        "test": {"rows": int(len(test)), "columns": int(test.shape[1]),
                 "missing_values": int(test.isna().sum().sum()),
                 "duplicate_feature_rows": int(test.duplicated(subset=config.FEATURES).sum()),
                 "feature_ranges": _ranges(test)},
        "flood": {"rows": int(len(flood)), "columns": int(flood.shape[1]),
                  "missing_values": int(flood.isna().sum().sum()),
                  "duplicate_feature_rows": int(flood.duplicated(subset=config.FEATURES).sum())},
        "feature_stats": {f: {k: float(desc.loc[f, k]) for k in ["min", "25%", "50%", "75%", "max", "mean", "std"]}
                          for f in config.FEATURES},
        "target_stats": {k: float(v) for k, v in train[config.TARGET].describe().items()},
        "target_unique_values": int(train[config.TARGET].nunique()),
        "correlation": {"columns": cols, "matrix": corr.round(6).values.tolist()},
    }


def main() -> None:
    config.REPORTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    metrics = json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))
    preds = pd.read_csv(config.VALIDATION_PREDICTIONS_PATH)
    best = metrics["best_model"]

    recomputed = regression_metrics(preds["actual"], preds["predicted"])
    for key in ("MAE", "RMSE", "R2"):
        assert abs(recomputed[key] - metrics["models"][best][key]) < 1e-9, f"{key} mismatch"
    print(f"Metrics recomputed from saved validation predictions match metrics.json ({best}).")

    train, test, flood = dp.load_train(), dp.load_test(), dp.load_flood_external()
    summary = build_data_summary(train, test, flood)
    residual = preds["actual"] - preds["predicted"]
    summary["validation_residuals"] = {
        "mean": float(residual.mean()), "std": float(residual.std()),
        "within_0.02": float((residual.abs() <= 0.02).mean()),
        "within_0.05": float((residual.abs() <= 0.05).mean()),
        "predicted_min": float(preds["predicted"].min()), "predicted_max": float(preds["predicted"].max()),
    }
    DATA_SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    sample = train.sample(n=min(SAMPLE_ROWS, len(train)), random_state=config.RANDOM_SEED)
    sample.to_csv(TRAIN_SAMPLE_PATH, index=False)

    corr = train[config.FEATURES + [config.TARGET]].corr()
    importance = pd.read_csv(config.FEATURE_IMPORTANCE_PATH)
    figures = {
        "target_distribution.png": charts.plot_target_distribution(train[config.TARGET]),
        "feature_distributions.png": charts.plot_feature_distributions(sample),
        "correlation_heatmap.png": charts.plot_correlation_heatmap(corr),
        "model_comparison.png": charts.plot_metric_comparison(metrics["models"]),
        "actual_vs_predicted.png": charts.plot_actual_vs_predicted(preds["actual"], preds["predicted"]),
        "residuals.png": charts.plot_residuals(preds["actual"], preds["predicted"]),
        "feature_importance.png": charts.plot_feature_importance(importance),
    }
    for name, fig in figures.items():
        fig.savefig(FIGURES_DIR / name, dpi=130)
        print("Saved", FIGURES_DIR / name)
    print("Residual summary:", {k: round(v, 5) for k, v in summary["validation_residuals"].items()})
    print("Test feature ranges exceed train?", {
        f: (summary["test"]["feature_ranges"][f], summary["train"]["feature_ranges"][f])
        for f in config.FEATURES
        if summary["test"]["feature_ranges"][f] != summary["train"]["feature_ranges"][f]})


if __name__ == "__main__":
    main()
