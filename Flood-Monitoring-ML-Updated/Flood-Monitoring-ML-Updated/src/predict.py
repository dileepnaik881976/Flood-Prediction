"""Shared inference code used by the dashboard and the command line.

Command line (submission file for test.csv):
    python -m src.predict --input data/test.csv --output submissions/submission.csv ^
        --template data/sample_submission.csv

Design notes
- The saved model and feature schema are loaded once (cached) and never retrained here.
- Feature order always comes from models/feature_schema.json, so column order in an uploaded
  file does not matter.
- Predictions are NOT clipped. The model output is a regression estimate of the dataset's
  FloodProbability score; clipping would hide model behaviour.
- Two kinds of "outside the training data" are detected:
    1. per-feature range: one feature is beyond the min/max seen in training;
    2. joint range: the SUM of the 20 features is outside the range seen in training
       (schema key `feature_sum_range`, 63-144 for the shipped model). A linear model keeps
       adding ~0.0057 per feature point, so an input where every feature is high (e.g. all 10)
       is far outside the data and its output can leave the training target range (~0.285-0.725)
       and even exceed 1. Such rows are still predicted by default but flagged as extrapolations.
  Strict mode rejects them instead. Single predictions use warn-and-show (per-feature range is
  still enforced); batch predictions can use either mode.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from . import config

FLAG_COLUMN = "extrapolation_warning"


class PredictionInputError(ValueError):
    """Raised when input data cannot be used for prediction."""


# ----------------------------------------------------------------------------- artifacts
@lru_cache(maxsize=1)
def load_schema() -> dict:
    if not config.SCHEMA_PATH.exists():
        raise FileNotFoundError(
            f"Missing {config.SCHEMA_PATH}. Train the model first:  python -m src.train")
    schema = json.loads(config.SCHEMA_PATH.read_text(encoding="utf-8"))
    if schema["features"] != config.FEATURES:
        raise PredictionInputError("Saved feature schema does not match src/config.py FEATURES.")
    sum_range = schema.get("feature_sum_range")
    if not (isinstance(sum_range, list) and len(sum_range) == 2 and sum_range[0] <= sum_range[1]):
        raise PredictionInputError(
            "feature_schema.json has no valid 'feature_sum_range'. Run  python -m src.update_schema  "
            "(no retraining needed) or  python -m src.train.")
    return schema


@lru_cache(maxsize=1)
def load_model():
    if not config.MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing {config.MODEL_PATH}. Train the model first:  python -m src.train")
    model = joblib.load(config.MODEL_PATH)
    expected = getattr(model, "feature_names_in_", None)
    if expected is not None and list(expected) != load_schema()["features"]:
        raise PredictionInputError("Model feature order differs from the saved schema.")
    return model


# ----------------------------------------------------------------------------- validation
def _empty_mask() -> np.ndarray:
    return np.zeros(0, dtype=bool)


@dataclass
class ValidationReport:
    n_rows: int = 0
    blocking_errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    n_valid: int = 0
    n_outside_training_range: int = 0      # rows with at least one feature beyond its own min/max
    n_outside_joint_range: int = 0         # rows whose feature SUM is outside the training sum range
    outside_joint_mask: np.ndarray = field(default_factory=_empty_mask)   # per input row
    outside_any_mask: np.ndarray = field(default_factory=_empty_mask)     # per-feature OR joint
    rejected: pd.DataFrame = field(default_factory=lambda: pd.DataFrame(columns=["row", "reason"]))

    @property
    def ok(self) -> bool:
        return not self.blocking_errors


def validate_frame(df: pd.DataFrame, strict_range: bool = False, strict_joint: bool | None = None):
    """Validate a dataframe of features. Returns (numeric_features_or_None, valid_mask, report).

    Blocking errors: missing feature columns, empty file.
    Row-level rejections: missing, non-numeric, infinite or negative values; and, only in strict
    mode, values outside the per-feature training range (strict_range) or a feature sum outside
    the training sum range (strict_joint, which defaults to the value of strict_range).
    """
    if strict_joint is None:
        strict_joint = strict_range
    report = ValidationReport(n_rows=len(df))
    schema = load_schema()
    features, ranges = schema["features"], schema["feature_ranges"]
    sum_low, sum_high = schema["feature_sum_range"]

    if len(df) == 0:
        report.blocking_errors.append("The file contains no rows.")
        return None, np.zeros(0, dtype=bool), report
    missing = [c for c in features if c not in df.columns]
    if missing:
        report.blocking_errors.append(f"Missing required columns: {missing}")
        return None, np.zeros(len(df), dtype=bool), report
    unexpected = [c for c in df.columns if c not in features and c != config.ID_COL]
    if unexpected:
        report.warnings.append(f"Ignoring unexpected columns: {unexpected}")
    if config.ID_COL in df.columns and df[config.ID_COL].duplicated().any():
        report.warnings.append(f"Column '{config.ID_COL}' contains duplicate values.")

    raw = df[features]
    numeric = raw.apply(pd.to_numeric, errors="coerce").astype("float64")
    was_missing = raw.isna()
    non_numeric = numeric.isna() & ~was_missing
    infinite = pd.DataFrame(np.isinf(numeric.to_numpy()), index=numeric.index, columns=features)
    negative = numeric < 0
    lows = pd.Series({f: ranges[f]["min"] for f in features})
    highs = pd.Series({f: ranges[f]["max"] for f in features})
    outside = (numeric.lt(lows, axis=1) | numeric.gt(highs, axis=1)) & numeric.notna() & ~infinite
    outside_feature = outside.any(axis=1)

    # Joint range: NaN sums (missing values) compare False and are rejected for other reasons.
    feature_sum = numeric.sum(axis=1, skipna=False)
    outside_joint = (feature_sum < sum_low) | (feature_sum > sum_high)
    report.outside_joint_mask = outside_joint.to_numpy()
    report.outside_any_mask = (outside_feature | outside_joint).to_numpy()
    report.n_outside_training_range = int(outside_feature.sum())
    report.n_outside_joint_range = int(outside_joint.sum())

    checks = [(was_missing.any(axis=1), "missing value"),
              (non_numeric.any(axis=1), "non-numeric value"),
              (infinite.any(axis=1), "infinite value"),
              (negative.any(axis=1), "negative value")]
    if strict_range:
        checks.append((outside_feature, "value outside training range"))
    if strict_joint:
        checks.append((outside_joint, "feature sum outside training range"))
    reason = pd.Series("", index=df.index)
    for mask, label in checks:
        reason = reason.where(~mask, reason + label + "; ")
    valid_mask = (reason == "").to_numpy()

    if (~valid_mask).any():
        rejected = pd.DataFrame({"row": np.flatnonzero(~valid_mask) + 1,
                                 "reason": reason[~valid_mask].str.rstrip("; ").to_numpy()})
        if config.ID_COL in df.columns:
            rejected.insert(1, config.ID_COL, df[config.ID_COL].to_numpy()[~valid_mask])
        report.rejected = rejected
    report.n_valid = int(valid_mask.sum())
    if report.n_outside_training_range and not strict_range:
        report.warnings.append(
            f"{report.n_outside_training_range:,} row(s) contain a feature value outside the range seen in "
            "training; predictions for them are extrapolations and less trustworthy.")
    if report.n_outside_joint_range and not strict_joint:
        report.warnings.append(
            f"{report.n_outside_joint_range:,} row(s) have a feature sum outside the range seen in training "
            f"({sum_low}-{sum_high}). The model has never seen such combinations, so these predictions are "
            "extrapolations and can fall far outside the FloodProbability range seen in training.")
    return numeric, valid_mask, report


# ----------------------------------------------------------------------------- prediction
def predict_numeric(features_df: pd.DataFrame, chunk_size: int = 200_000) -> np.ndarray:
    """Predict from an already validated numeric frame, enforcing schema column order."""
    model = load_model()
    ordered = features_df[load_schema()["features"]]
    out = np.empty(len(ordered), dtype="float64")
    for start in range(0, len(ordered), chunk_size):
        out[start:start + chunk_size] = model.predict(ordered.iloc[start:start + chunk_size])
    if not np.isfinite(out).all():
        raise PredictionInputError("Model produced non-finite predictions.")
    return out


@dataclass(frozen=True)
class SinglePrediction:
    value: float                     # raw model output, never clipped
    feature_sum: float
    sum_range: tuple                 # (min, max) feature sum seen in training
    outside_joint_range: bool        # True -> extrapolation: show a warning with the value


def predict_single_with_checks(values: dict) -> SinglePrediction:
    """Predict one row from {feature: value}.

    All 20 features are required and each must be within its training range (the dashboard sliders
    enforce this). A feature sum outside the training sum range does NOT block the prediction
    (warn-and-show); it is reported in `outside_joint_range`.
    """
    df = pd.DataFrame([values])
    numeric, mask, report = validate_frame(df, strict_range=True, strict_joint=False)
    if not report.ok:
        raise PredictionInputError("; ".join(report.blocking_errors))
    if not mask.all():
        raise PredictionInputError(f"Invalid input: {report.rejected['reason'].iloc[0]}")
    schema = load_schema()
    return SinglePrediction(
        value=float(predict_numeric(numeric)[0]),
        feature_sum=float(numeric.sum(axis=1).iloc[0]),
        sum_range=tuple(schema["feature_sum_range"]),
        outside_joint_range=bool(report.outside_joint_mask[0]),
    )


def predict_single(values: dict) -> float:
    """Convenience wrapper returning only the raw prediction (see predict_single_with_checks)."""
    return predict_single_with_checks(values).value


def predict_batch(df: pd.DataFrame, strict_range: bool = False, strict_joint: bool | None = None,
                  include_flags: bool = False):
    """Predict for every valid row. Returns (result_df, report).

    result_df has the input 'id' column when present (otherwise 'row', the 1-based position in the
    uploaded file) followed by FloodProbability, in the original row order. With include_flags=True
    an extra boolean column `extrapolation_warning` marks rows whose per-feature values or feature
    sum are outside the training range. The default output format is unchanged.
    """
    numeric, mask, report = validate_frame(df, strict_range=strict_range, strict_joint=strict_joint)
    if not report.ok:
        return pd.DataFrame(columns=[config.ID_COL, config.TARGET]), report
    preds = predict_numeric(numeric.loc[mask])
    if config.ID_COL in df.columns:
        ids = df.loc[mask, config.ID_COL].reset_index(drop=True)
        result = pd.DataFrame({config.ID_COL: ids, config.TARGET: preds})
    else:
        result = pd.DataFrame({"row": np.flatnonzero(mask) + 1, config.TARGET: preds})
    if include_flags:
        result[FLAG_COLUMN] = report.outside_any_mask[mask]
    return result, report


def linear_contributions(values: dict, reference_means: dict):
    """For linear models only: per-feature effect on the prediction relative to an average input."""
    model = load_model()
    coef = getattr(model, "coef_", None)
    if coef is None or np.ndim(coef) != 1:
        return None
    features = load_schema()["features"]
    return pd.Series({f: float(c) * (values[f] - reference_means[f]) for f, c in zip(features, coef)})


# ----------------------------------------------------------------------------- submission CLI
def generate_submission(input_path: Path, output_path: Path, template_path: Path | None = None,
                        chunksize: int = 200_000) -> dict:
    """Write id,FloodProbability for every row of input_path (format unchanged, no flag column)."""
    input_path, output_path = Path(input_path), Path(output_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        output_path.unlink()

    total, first = 0, True
    lo, hi, running_sum = np.inf, -np.inf, 0.0
    outside_joint = outside_feature = 0
    for chunk in pd.read_csv(input_path, chunksize=chunksize):
        if config.ID_COL not in chunk.columns:
            raise PredictionInputError(f"'{config.ID_COL}' column is required to build a submission.")
        result, report = predict_batch(chunk, strict_range=False)
        if not report.ok:
            raise PredictionInputError("; ".join(report.blocking_errors))
        if len(report.rejected):
            raise PredictionInputError(
                f"{len(report.rejected)} invalid row(s) in a chunk, e.g. {report.rejected.head(3).to_dict('records')}")
        outside_joint += report.n_outside_joint_range
        outside_feature += report.n_outside_training_range
        result.to_csv(output_path, mode="w" if first else "a", header=first, index=False)
        first = False
        total += len(result)
        lo, hi = min(lo, result[config.TARGET].min()), max(hi, result[config.TARGET].max())
        running_sum += float(result[config.TARGET].sum())

    summary = {"rows": total, "min": float(lo), "max": float(hi), "mean": running_sum / total,
               "rows_with_feature_outside_training_range": outside_feature,
               "rows_with_feature_sum_outside_training_range": outside_joint,
               "output": str(output_path)}
    if template_path is not None:
        summary["template_check"] = verify_against_template(output_path, template_path)
    return summary


def verify_against_template(submission_path: Path, template_path: Path) -> dict:
    sub = pd.read_csv(submission_path)
    tpl = pd.read_csv(template_path)
    checks = {
        "same_columns": list(sub.columns) == list(tpl.columns),
        "same_row_count": len(sub) == len(tpl),
        "ids_identical_and_in_order": bool((sub[config.ID_COL].to_numpy() == tpl[config.ID_COL].to_numpy()).all())
        if len(sub) == len(tpl) else False,
        "all_predictions_finite": bool(np.isfinite(sub[config.TARGET].to_numpy()).all()),
    }
    checks["passed"] = all(checks.values())
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate FloodProbability predictions for a CSV.")
    parser.add_argument("--input", default=str(config.TEST_CSV))
    parser.add_argument("--output", default=str(config.ROOT / "submissions" / "submission.csv"))
    parser.add_argument("--template", default=str(config.SAMPLE_SUBMISSION_CSV),
                        help="sample_submission.csv used to verify ids/columns (use '' to skip)")
    parser.add_argument("--chunksize", type=int, default=200_000)
    args = parser.parse_args()
    template = Path(args.template) if args.template else None
    summary = generate_submission(Path(args.input), Path(args.output), template, args.chunksize)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
