"""Loading, validating and splitting the flood datasets."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from . import config


class SchemaError(ValueError):
    """Raised when a dataframe does not match the expected feature schema."""


def load_csv(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}. Put the CSV files in the data/ folder.")
    return pd.read_csv(path)


def validate_features(df: pd.DataFrame, require_target: bool = False) -> None:
    """Check required columns exist, are numeric, and have no missing values."""
    missing = [c for c in config.FEATURES if c not in df.columns]
    if missing:
        raise SchemaError(f"Missing feature columns: {missing}")
    if require_target and config.TARGET not in df.columns:
        raise SchemaError(f"Missing target column: {config.TARGET}")
    cols = config.FEATURES + ([config.TARGET] if require_target else [])
    non_numeric = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise SchemaError(f"Non-numeric columns: {non_numeric}")
    if df[cols].isna().any().any():
        bad = df[cols].columns[df[cols].isna().any()].tolist()
        raise SchemaError(f"Missing values found in: {bad}")
    if not np.isfinite(df[cols].to_numpy(dtype=float)).all():
        raise SchemaError("Infinite values found in the data.")


def load_train() -> pd.DataFrame:
    df = load_csv(config.TRAIN_CSV)
    validate_features(df, require_target=True)
    return df


def load_test() -> pd.DataFrame:
    df = load_csv(config.TEST_CSV)
    validate_features(df)
    return df


def load_flood_external() -> pd.DataFrame:
    df = load_csv(config.FLOOD_CSV)
    validate_features(df, require_target=True)
    return df


def split_train_validation(df: pd.DataFrame):
    """Reproducible split. Returns X_train, X_val, y_train, y_val (id excluded)."""
    X = df[config.FEATURES]
    y = df[config.TARGET]
    return train_test_split(
        X, y, test_size=config.VALIDATION_FRACTION, random_state=config.RANDOM_SEED
    )


def feature_ranges(df: pd.DataFrame) -> dict:
    """Observed min/max per feature, used to set sensible input widgets."""
    return {c: {"min": int(df[c].min()), "max": int(df[c].max())} for c in config.FEATURES}


def feature_sum_range(df: pd.DataFrame) -> list:
    """[min, max] of the row-wise sum of the 20 features seen in training data.

    Per-feature ranges cannot detect an input where *all* features are high at once; this joint
    range can (see src/predict.py).
    """
    totals = df[config.FEATURES].sum(axis=1)
    return [int(totals.min()), int(totals.max())]
