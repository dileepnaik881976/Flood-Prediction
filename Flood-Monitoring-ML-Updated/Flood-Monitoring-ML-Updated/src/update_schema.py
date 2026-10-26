"""Add the training feature-sum range to an existing models/feature_schema.json WITHOUT retraining.

Run from the project root:   python -m src.update_schema
Only the schema JSON is rewritten; the model file is never touched. Safe to run repeatedly.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import config, data_processing as dp


def update_schema(train_df: pd.DataFrame, schema_path: Path = config.SCHEMA_PATH) -> dict:
    schema_path = Path(schema_path)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    schema["feature_sum_range"] = dp.feature_sum_range(train_df)
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    return schema


def main() -> None:
    schema = update_schema(dp.load_train())
    print("feature_sum_range =", schema["feature_sum_range"], "->", config.SCHEMA_PATH)


if __name__ == "__main__":
    main()
