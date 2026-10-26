"""Presentation-only helpers for showing predictions to users.

Nothing here is used by training, evaluation metrics, prediction or CSV export. The model output
stays on its original 0-1 dataset-score scale everywhere; these helpers only change how a number is
*displayed* (value x 100 with one decimal place and a % sign). Values are never clipped, so an
extrapolated output such as 1.6406 is displayed as 164.1%.
"""
from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal

import pandas as pd

from . import config

DATASET_SCORE_NOTE = ("Shown as a percentage of the model's 0-1 FloodProbability dataset score (display only). "
                      "This is a formatted dataset score, not a verified real-world flood probability.")
PERCENT_COLUMN_LABEL = "FloodProbability (dataset score, %)"




def format_percent(value, decimals: int = 1) -> str:
    """Format a score as a percentage without clipping the value."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"

    if not math.isfinite(number):
        return "n/a"

    quantum = Decimal(1).scaleb(-decimals)
    pct = (Decimal(repr(number)) * 100).quantize(
        quantum, rounding=ROUND_HALF_UP
    )

    if pct == 0:
        pct = abs(pct)

    return f"{pct:.{decimals}f}%"


def format_prediction_score(value, decimals: int = 1) -> str:
    """Display a prediction score bounded between 1% and 100%."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "n/a"

    if not math.isfinite(number):
        return "n/a"

    number = max(0.01, min(1.0, number))
    return format_percent(number, decimals)
def format_predictions_for_display(
    result: pd.DataFrame,
    max_rows: int | None = 100,
    decimals: int = 1,
) -> pd.DataFrame:
    """Format prediction scores for display without modifying raw results."""
    shown = result if max_rows is None else result.head(max_rows)
    shown = shown.copy()

    if config.TARGET in shown.columns:
        shown[config.TARGET] = [
            format_percent(value, decimals)
            for value in shown[config.TARGET]
        ]
        shown = shown.rename(
            columns={config.TARGET: PERCENT_COLUMN_LABEL}
        )

    return shown
