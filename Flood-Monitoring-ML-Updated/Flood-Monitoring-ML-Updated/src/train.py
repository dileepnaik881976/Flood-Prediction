"""Train and compare regression models. Run from the project root:  python -m src.train"""
from __future__ import annotations

import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from . import config, data_processing as dp


def regression_metrics(y_true, y_pred) -> dict:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred)),
    }


def build_models() -> dict:
    seed = config.RANDOM_SEED
    return {
        "Baseline (mean)": (DummyRegressor(strategy="mean"), False),
        "Linear Regression": (LinearRegression(), False),
        "Random Forest": (RandomForestRegressor(
            n_estimators=60, min_samples_leaf=5, max_features=0.5, n_jobs=-1, random_state=seed), True),
        "Extra Trees": (ExtraTreesRegressor(
            n_estimators=60, min_samples_leaf=5, max_features=0.5, n_jobs=-1, random_state=seed), True),
        "HistGradientBoosting": (HistGradientBoostingRegressor(
            max_iter=300, learning_rate=0.1, random_state=seed), False),
    }


def build_schema(best_name: str, train_df: pd.DataFrame) -> dict:
    """Feature schema saved next to the model (feature order, per-feature and feature-sum ranges)."""
    return {
        "features": config.FEATURES, "target": config.TARGET, "model_name": best_name,
        "feature_ranges": dp.feature_ranges(train_df),
        "feature_sum_range": dp.feature_sum_range(train_df),
        "random_seed": config.RANDOM_SEED,
    }


def main() -> None:
    config.MODELS_DIR.mkdir(exist_ok=True)
    config.REPORTS_DIR.mkdir(exist_ok=True)

    df = dp.load_train()
    X_train, X_val, y_train, y_val = dp.split_train_validation(df)
    print(f"Train rows: {len(X_train):,} | Validation rows: {len(X_val):,}")

    sub_idx = X_train.sample(
        n=min(config.TREE_ENSEMBLE_TRAIN_ROWS, len(X_train)), random_state=config.RANDOM_SEED).index

    results, fitted, val_preds = {}, {}, {}
    for name, (model, use_subsample) in build_models().items():
        Xf, yf = (X_train.loc[sub_idx], y_train.loc[sub_idx]) if use_subsample else (X_train, y_train)
        start = time.time()
        model.fit(Xf, yf)
        seconds = time.time() - start
        pred = model.predict(X_val)
        metrics = regression_metrics(y_val, pred)
        metrics.update({"train_rows_used": int(len(Xf)), "fit_seconds": round(seconds, 1)})
        results[name], fitted[name], val_preds[name] = metrics, model, pred
        print(f"{name:22s} MAE={metrics['MAE']:.5f} RMSE={metrics['RMSE']:.5f} "
              f"R2={metrics['R2']:.5f} ({seconds:.1f}s, {len(Xf):,} rows)", flush=True)

    candidates = {k: v for k, v in results.items() if k != "Baseline (mean)"}
    best_name = min(candidates, key=lambda k: candidates[k]["RMSE"])  # selected on validation RMSE
    best_model = fitted[best_name]
    print(f"\nBest model on validation RMSE: {best_name}")

    # Permutation importance on a validation subsample (model-agnostic).
    imp_idx = X_val.sample(n=min(20_000, len(X_val)), random_state=config.RANDOM_SEED).index
    imp = permutation_importance(best_model, X_val.loc[imp_idx], y_val.loc[imp_idx],
                                 n_repeats=3, random_state=config.RANDOM_SEED, n_jobs=-1,
                                 scoring="neg_root_mean_squared_error")
    pd.DataFrame({"feature": config.FEATURES, "importance_mean": imp.importances_mean,
                  "importance_std": imp.importances_std}
                 ).sort_values("importance_mean", ascending=False).to_csv(config.FEATURE_IMPORTANCE_PATH, index=False)

    # Separate external check on flood.csv (never used for training or selection).
    external = None
    try:
        ext = dp.load_flood_external()
        external = regression_metrics(ext[config.TARGET], best_model.predict(ext[config.FEATURES]))
        print("External check on flood.csv:", {k: round(v, 5) for k, v in external.items()})
    except Exception as exc:  # keep training usable even if flood.csv is absent
        print(f"Skipping external check: {exc}")

    joblib.dump(best_model, config.MODEL_PATH, compress=3)
    config.SCHEMA_PATH.write_text(json.dumps(build_schema(best_name, df), indent=2), encoding="utf-8")

    pd.DataFrame({"actual": y_val.to_numpy(), "predicted": val_preds[best_name]}
                 ).to_csv(config.VALIDATION_PREDICTIONS_PATH, index=False)
    pd.DataFrame({name: p for name, p in val_preds.items()}).assign(actual=y_val.to_numpy()
        ).sample(n=min(50_000, len(y_val)), random_state=config.RANDOM_SEED
        ).to_csv(config.REPORTS_DIR / "validation_predictions_all_models.csv", index=False)

    config.METRICS_PATH.write_text(json.dumps({
        "best_model": best_name, "models": results, "external_flood_csv": external,
        "split": {"train_rows": int(len(X_train)), "validation_rows": int(len(X_val)),
                  "seed": config.RANDOM_SEED, "validation_fraction": config.VALIDATION_FRACTION},
    }, indent=2), encoding="utf-8")
    print("Saved model, schema, metrics and validation predictions.")


if __name__ == "__main__":
    main()
