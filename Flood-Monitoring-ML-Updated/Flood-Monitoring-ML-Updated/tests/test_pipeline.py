"""Pipeline tests (standard library unittest; pytest can also run them).

Run from the project root:   python -m unittest discover -s tests -v
"""
import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src import config, data_processing as dp, predict
from src.train import build_models, regression_metrics

HAVE_DATA = all(p.exists() for p in [config.TRAIN_CSV, config.TEST_CSV, config.SAMPLE_SUBMISSION_CSV, config.FLOOD_CSV])


def mid_row(value=5):
    return {f: value for f in config.FEATURES}


class TestDataLoading(unittest.TestCase):
    @unittest.skipUnless(HAVE_DATA, "CSV files not in data/")
    def test_datasets_load_with_expected_shapes(self):
        train, test = dp.load_train(), dp.load_test()
        self.assertEqual(train.shape, (1117957, 22))
        self.assertEqual(test.shape, (745305, 21))
        self.assertEqual(dp.load_flood_external().shape, (50000, 21))
        self.assertNotIn(config.ID_COL, config.FEATURES)

    @unittest.skipUnless(HAVE_DATA, "CSV files not in data/")
    def test_split_is_reproducible_and_excludes_id(self):
        df = pd.read_csv(config.TRAIN_CSV, nrows=5000)
        a = dp.split_train_validation(df)
        b = dp.split_train_validation(df)
        self.assertTrue(a[1].index.equals(b[1].index))
        self.assertEqual(list(a[0].columns), config.FEATURES)
        self.assertEqual(len(a[0]) + len(a[1]), len(df))
        self.assertTrue(set(a[0].index).isdisjoint(a[1].index))


class TestSchemaValidation(unittest.TestCase):
    def frame(self):
        return pd.DataFrame([mid_row()] * 3).assign(**{config.TARGET: 0.5})

    def test_valid_frame_passes(self):
        dp.validate_features(self.frame(), require_target=True)

    def test_missing_column_raises(self):
        with self.assertRaises(dp.SchemaError):
            dp.validate_features(self.frame().drop(columns=["Landslides"]))

    def test_non_numeric_raises(self):
        df = self.frame().astype({"Landslides": object})
        df.loc[0, "Landslides"] = "abc"
        with self.assertRaises(dp.SchemaError):
            dp.validate_features(df)

    def test_missing_value_raises(self):
        df = self.frame()
        df.loc[1, "Siltation"] = np.nan
        with self.assertRaises(dp.SchemaError):
            dp.validate_features(df)


class TestTrainingSmall(unittest.TestCase):
    @unittest.skipUnless(HAVE_DATA, "CSV files not in data/")
    def test_small_training_and_artifact_roundtrip(self):
        df = pd.read_csv(config.TRAIN_CSV, nrows=20000)
        X_tr, X_val, y_tr, y_val = dp.split_train_validation(df)
        model = build_models()["Linear Regression"][0].fit(X_tr, y_tr)
        metrics = regression_metrics(y_val, model.predict(X_val))
        self.assertGreater(metrics["R2"], 0.8)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.joblib"
            joblib.dump(model, path)
            loaded = joblib.load(path)
            np.testing.assert_allclose(loaded.predict(X_val), model.predict(X_val))


class TestSavedArtifacts(unittest.TestCase):
    def test_artifacts_exist_and_match_config(self):
        self.assertTrue(config.MODEL_PATH.exists())
        schema = json.loads(config.SCHEMA_PATH.read_text(encoding="utf-8"))
        self.assertEqual(schema["features"], config.FEATURES)
        self.assertEqual(set(schema["feature_ranges"]), set(config.FEATURES))
        model = predict.load_model()
        self.assertIsInstance(model, LinearRegression if schema["model_name"] == "Linear Regression" else object)
        self.assertEqual(list(model.feature_names_in_), config.FEATURES)

    def test_saved_metrics_are_sane(self):
        m = json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))
        best = m["models"][m["best_model"]]
        self.assertGreater(best["R2"], m["models"]["Baseline (mean)"]["R2"])
        self.assertLessEqual(best["RMSE"], min(v["RMSE"] for k, v in m["models"].items() if k != "Baseline (mean)"))


class TestSinglePrediction(unittest.TestCase):
    def test_prediction_is_finite_float_and_deterministic(self):
        a, b = predict.predict_single(mid_row()), predict.predict_single(mid_row())
        self.assertIsInstance(a, float)
        self.assertTrue(np.isfinite(a))
        self.assertEqual(a, b)
        self.assertTrue(0.2 < a < 0.8)

    def test_higher_inputs_give_higher_score(self):
        self.assertGreater(predict.predict_single(mid_row(8)), predict.predict_single(mid_row(3)))

    def test_missing_feature_rejected(self):
        row = mid_row()
        del row["Landslides"]
        with self.assertRaises(predict.PredictionInputError):
            predict.predict_single(row)

    def test_invalid_values_rejected(self):
        for bad in (-1, np.nan, "abc", 999):
            row = mid_row()
            row["Landslides"] = bad
            with self.assertRaises(predict.PredictionInputError, msg=str(bad)):
                predict.predict_single(row)


class TestFeatureOrder(unittest.TestCase):
    def test_column_order_does_not_change_predictions(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame(rng.integers(0, 10, size=(50, 20)), columns=config.FEATURES)
        shuffled = df[list(reversed(config.FEATURES))]
        a, _ = predict.predict_batch(df)
        b, _ = predict.predict_batch(shuffled)
        np.testing.assert_allclose(a[config.TARGET], b[config.TARGET])


class TestBatchPrediction(unittest.TestCase):
    def make(self, n=10):
        rng = np.random.default_rng(1)
        df = pd.DataFrame(rng.integers(0, 10, size=(n, 20)), columns=config.FEATURES)
        df.insert(0, config.ID_COL, np.arange(100, 100 + n))
        return df

    def test_output_format_and_id_preserved(self):
        df = self.make()
        result, report = predict.predict_batch(df)
        self.assertTrue(report.ok)
        self.assertEqual(list(result.columns), [config.ID_COL, config.TARGET])
        self.assertEqual(result[config.ID_COL].tolist(), df[config.ID_COL].tolist())
        self.assertTrue(np.isfinite(result[config.TARGET]).all())
        self.assertEqual(len(result), len(df))

    def test_no_id_column_uses_row_numbers(self):
        result, _ = predict.predict_batch(self.make().drop(columns=[config.ID_COL]))
        self.assertEqual(list(result.columns), ["row", config.TARGET])
        self.assertEqual(result["row"].tolist(), list(range(1, 11)))

    def test_missing_columns_is_blocking_error(self):
        result, report = predict.predict_batch(self.make().drop(columns=["Siltation", "Landslides"]))
        self.assertFalse(report.ok)
        self.assertIn("Missing required columns", report.blocking_errors[0])
        self.assertEqual(len(result), 0)

    def test_unexpected_columns_warn_and_are_ignored(self):
        df = self.make().assign(notes="x")
        result, report = predict.predict_batch(df)
        self.assertTrue(report.ok)
        self.assertTrue(any("unexpected" in w for w in report.warnings))
        self.assertEqual(len(result), len(df))

    def test_bad_rows_are_rejected_with_reasons_and_others_predicted(self):
        df = self.make().astype({"Landslides": object})
        df.loc[1, "Landslides"] = "abc"          # non-numeric
        df.loc[2, "Siltation"] = np.nan          # missing
        df.loc[3, "Urbanization"] = -4           # negative
        result, report = predict.predict_batch(df)
        self.assertEqual(report.n_valid, 7)
        self.assertEqual(sorted(report.rejected["row"]), [2, 3, 4])
        reasons = " ".join(report.rejected["reason"])
        for text in ("non-numeric", "missing", "negative"):
            self.assertIn(text, reasons)
        self.assertEqual(result[config.ID_COL].tolist(), [100, 104, 105, 106, 107, 108, 109])

    def test_out_of_range_is_warning_by_default_and_rejected_in_strict_mode(self):
        df = self.make()
        df.loc[0, "WetlandLoss"] = 60
        result, report = predict.predict_batch(df)
        self.assertEqual(len(result), 10)
        self.assertEqual(report.n_outside_training_range, 1)
        strict_result, strict_report = predict.predict_batch(df, strict_range=True)
        self.assertEqual(len(strict_result), 9)
        self.assertIn("outside training range", strict_report.rejected["reason"].iloc[0])

    def test_empty_file_is_blocking_error(self):
        _, report = predict.predict_batch(pd.DataFrame(columns=config.FEATURES))
        self.assertFalse(report.ok)

    def test_chunked_prediction_matches_unchunked(self):
        df = self.make(30)
        numeric, _, _ = predict.validate_frame(df)
        np.testing.assert_allclose(predict.predict_numeric(numeric, chunk_size=7),
                                   predict.predict_numeric(numeric, chunk_size=1000))


class TestSubmission(unittest.TestCase):
    @unittest.skipUnless(HAVE_DATA, "CSV files not in data/")
    def test_submission_generation_on_head_of_test(self):
        with tempfile.TemporaryDirectory() as tmp:
            small_in = Path(tmp) / "small_test.csv"
            small_tpl = Path(tmp) / "small_sample.csv"
            out = Path(tmp) / "sub.csv"
            pd.read_csv(config.TEST_CSV, nrows=1000).to_csv(small_in, index=False)
            pd.read_csv(config.SAMPLE_SUBMISSION_CSV, nrows=1000).to_csv(small_tpl, index=False)
            summary = predict.generate_submission(small_in, out, small_tpl, chunksize=300)
            self.assertTrue(summary["template_check"]["passed"], summary)
            self.assertEqual(summary["rows"], 1000)

    @unittest.skipUnless((config.ROOT / "submissions" / "submission.csv").exists() and HAVE_DATA,
                         "full submission not generated yet")
    def test_full_submission_matches_template(self):
        checks = predict.verify_against_template(config.ROOT / "submissions" / "submission.csv",
                                                 config.SAMPLE_SUBMISSION_CSV)
        self.assertTrue(checks["passed"], checks)


if __name__ == "__main__":
    unittest.main()
