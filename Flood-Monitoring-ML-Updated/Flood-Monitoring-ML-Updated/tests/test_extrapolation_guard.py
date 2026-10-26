"""Regression tests for the joint-range (feature-sum) extrapolation guard.

Background: a user fed rows with all 20 features equal to 5, 10 and 15. Each value is inside the
per-feature training range, but the feature sums (100, 200, 300) are not: training sums span 63-144.
The linear model returned 0.5113, 1.0760 and 1.6406. That is expected linear extrapolation, not a bug,
so predictions must NOT change; they must be flagged instead.
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from src import config, data_processing as dp, predict, update_schema
from src.train import build_schema

F = config.FEATURES
FIXTURE = config.ROOT / "tests" / "fixtures" / "baseline_train_rows.csv"
SUBMISSION = config.ROOT / "submissions" / "submission.csv"
HAVE_DATA = all(p.exists() for p in [config.TRAIN_CSV, config.TEST_CSV, config.SAMPLE_SUBMISSION_CSV])


def uniform(v):
    return {f: v for f in F}


def sample_frame():
    df = pd.DataFrame([uniform(5), uniform(10), uniform(15)])
    df.insert(0, config.ID_COL, [1, 2, 3])
    return df


def frame_with_sum(total, base):
    """Row whose 20 features are `base` or `base+1` so that they add up to `total` exactly."""
    extra = total - base * len(F)
    assert 0 <= extra <= len(F)
    return pd.DataFrame([{f: base + (1 if i < extra else 0) for i, f in enumerate(F)}])


class TestSchemaSumRange(unittest.TestCase):
    def test_schema_has_valid_sum_range_matching_fixture_extremes(self):
        schema = json.loads(config.SCHEMA_PATH.read_text(encoding="utf-8"))
        lo, hi = schema["feature_sum_range"]
        self.assertIsInstance(lo, int)
        self.assertLess(lo, hi)
        fx = pd.read_csv(FIXTURE)   # contains the real min-sum and max-sum training rows
        self.assertEqual([lo, hi], [int(fx.feature_sum.min()), int(fx.feature_sum.max())])
        self.assertEqual([lo, hi], [63, 144])

    @unittest.skipUnless(HAVE_DATA, "CSV files not in data/")
    def test_schema_sum_range_equals_range_computed_from_train_csv(self):
        train = pd.read_csv(config.TRAIN_CSV, usecols=F)
        self.assertEqual(predict.load_schema()["feature_sum_range"], dp.feature_sum_range(train))

    def test_helper_and_build_schema_include_the_range(self):
        small = pd.DataFrame([uniform(2), uniform(3)])
        small[config.TARGET] = 0.5
        self.assertEqual(dp.feature_sum_range(small), [40, 60])
        schema = build_schema("Linear Regression", small)      # what python -m src.train writes
        self.assertEqual(schema["feature_sum_range"], [40, 60])
        self.assertEqual(schema["features"], F)
        self.assertIn("feature_ranges", schema)

    def test_update_schema_adds_key_preserves_rest_and_is_idempotent(self):
        original = json.loads(config.SCHEMA_PATH.read_text(encoding="utf-8"))
        old_style = {k: v for k, v in original.items() if k != "feature_sum_range"}
        small = pd.DataFrame([uniform(4), uniform(6)])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "schema.json"
            path.write_text(json.dumps(old_style), encoding="utf-8")
            first = update_schema.update_schema(small, path)
            second = update_schema.update_schema(small, path)
        self.assertEqual(first["feature_sum_range"], [80, 120])
        self.assertEqual(first, second)
        for key, value in old_style.items():
            self.assertEqual(first[key], value)

    def test_schema_without_sum_range_is_rejected_with_helpful_message(self):
        original = json.loads(config.SCHEMA_PATH.read_text(encoding="utf-8"))
        del original["feature_sum_range"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "schema.json"
            path.write_text(json.dumps(original), encoding="utf-8")
            predict.load_schema.cache_clear()
            try:
                with mock.patch.object(config, "SCHEMA_PATH", path):
                    with self.assertRaises(predict.PredictionInputError) as ctx:
                        predict.load_schema()
                self.assertIn("update_schema", str(ctx.exception))
            finally:
                predict.load_schema.cache_clear()
        self.assertEqual(predict.load_schema()["feature_sum_range"], [63, 144])


class TestReportedSampleInputs(unittest.TestCase):
    def test_predictions_are_unchanged_and_not_clipped(self):
        result, _ = predict.predict_batch(sample_frame())
        for got, expected in zip(result[config.TARGET], [0.5113, 1.0760, 1.6406]):
            self.assertAlmostEqual(got, expected, places=4)
        self.assertGreater(result[config.TARGET].iloc[2], 1.0)       # raw value, never clipped to [0, 1]

    def test_rows_with_out_of_domain_sums_are_flagged(self):
        result, report = predict.predict_batch(sample_frame(), include_flags=True)
        self.assertEqual(result[predict.FLAG_COLUMN].tolist(), [False, True, True])
        self.assertEqual(report.n_outside_joint_range, 2)
        # The per-feature check cannot see this problem (every value is within its own range).
        self.assertEqual(report.n_outside_training_range, 0)
        self.assertTrue(any("feature sum" in w for w in report.warnings))
        self.assertEqual(report.rejected.shape[0], 0)               # default mode: warn, still predict

    def test_default_output_format_is_unchanged_and_flags_are_opt_in(self):
        plain, _ = predict.predict_batch(sample_frame())
        flagged, _ = predict.predict_batch(sample_frame(), include_flags=True)
        self.assertEqual(list(plain.columns), [config.ID_COL, config.TARGET])
        self.assertEqual(list(flagged.columns), [config.ID_COL, config.TARGET, predict.FLAG_COLUMN])
        np.testing.assert_array_equal(plain[config.TARGET], flagged[config.TARGET])

    def test_flag_column_also_covers_single_feature_out_of_range(self):
        df = sample_frame().iloc[[0]].copy()
        df.loc[0, "WetlandLoss"] = 60
        result, report = predict.predict_batch(df, include_flags=True)
        self.assertTrue(bool(result[predict.FLAG_COLUMN].iloc[0]))
        self.assertEqual(report.n_outside_training_range, 1)

    def test_single_prediction_warns_but_still_returns_raw_value(self):
        ok = predict.predict_single_with_checks(uniform(5))
        self.assertFalse(ok.outside_joint_range)
        self.assertEqual(ok.feature_sum, 100)
        out = predict.predict_single_with_checks(uniform(10))
        self.assertTrue(out.outside_joint_range)
        self.assertEqual(out.sum_range, (63, 144))
        self.assertAlmostEqual(out.value, 1.0760, places=4)
        self.assertEqual(predict.predict_single(uniform(10)), out.value)   # wrapper, same value

    def test_single_prediction_still_rejects_invalid_values(self):
        row = uniform(5)
        row["Landslides"] = 999                                  # beyond that feature's own range
        with self.assertRaises(predict.PredictionInputError):
            predict.predict_single_with_checks(row)


class TestRealisticTrainingRows(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fx = pd.read_csv(FIXTURE)       # 200 random training rows + the min-sum and max-sum rows

    def test_in_range_rows_are_not_flagged_and_predictions_are_unchanged(self):
        result, report = predict.predict_batch(self.fx[[config.ID_COL] + F], include_flags=True)
        self.assertEqual(report.n_outside_joint_range, 0)
        self.assertEqual(report.n_outside_training_range, 0)
        self.assertFalse(result[predict.FLAG_COLUMN].any())
        # expected_prediction was recorded with the code BEFORE this change.
        np.testing.assert_allclose(result[config.TARGET], self.fx["expected_prediction"], rtol=0, atol=1e-12)

    def test_single_predictions_match_recorded_values(self):
        for _, row in self.fx.head(25).iterrows():
            out = predict.predict_single_with_checks({f: int(row[f]) for f in F})
            self.assertFalse(out.outside_joint_range)
            self.assertAlmostEqual(out.value, row["expected_prediction"], places=12)

    def test_extreme_real_rows_at_both_ends_of_the_range_are_not_flagged(self):
        self.assertEqual(sorted(self.fx.feature_sum)[0], 63)
        self.assertEqual(sorted(self.fx.feature_sum)[-1], 144)
        _, report = predict.predict_batch(self.fx[self.fx.feature_sum.isin([63, 144])][[config.ID_COL] + F])
        self.assertEqual(report.n_outside_joint_range, 0)

    def test_boundaries_are_inclusive_and_one_step_outside_is_flagged(self):
        cases = [(62, 3, True), (63, 3, False), (144, 7, False), (145, 7, True)]
        for total, base, expect_flag in cases:
            with self.subTest(total=total):
                _, report = predict.predict_batch(frame_with_sum(total, base))
                self.assertEqual(report.n_outside_joint_range, int(expect_flag))


class TestStrictMode(unittest.TestCase):
    def test_strict_rejects_out_of_domain_rows_and_keeps_ids(self):
        result, report = predict.predict_batch(sample_frame(), strict_range=True)
        self.assertEqual(result[config.ID_COL].tolist(), [1])
        self.assertEqual(report.rejected[config.ID_COL].tolist(), [2, 3])
        for reason in report.rejected["reason"]:
            self.assertIn("feature sum outside training range", reason)
        self.assertFalse(any("feature sum" in w for w in report.warnings))   # rejected, so no warning

    def test_joint_and_per_feature_strictness_are_independent(self):
        _, joint_only = predict.predict_batch(sample_frame(), strict_range=False, strict_joint=True)
        self.assertEqual(len(joint_only.rejected), 2)
        _, feature_only = predict.predict_batch(sample_frame(), strict_range=True, strict_joint=False)
        self.assertEqual(len(feature_only.rejected), 0)
        df = sample_frame().iloc[[0]].copy()
        df.loc[0, "WetlandLoss"] = 60
        _, rep = predict.predict_batch(df, strict_range=True, strict_joint=False)
        self.assertIn("value outside training range", rep.rejected["reason"].iloc[0])


@unittest.skipUnless(HAVE_DATA, "CSV files not in data/")
class TestTestCsvCompatibility(unittest.TestCase):
    def test_no_test_row_is_flagged_or_rejected_even_in_strict_joint_mode(self):
        flagged = rejected = total = 0
        for chunk in pd.read_csv(config.TEST_CSV, chunksize=200_000):
            _, report = predict.predict_batch(chunk, strict_range=False, strict_joint=True)
            flagged += report.n_outside_joint_range
            rejected += len(report.rejected)
            total += len(chunk)
        self.assertEqual(total, 745305)
        self.assertEqual(flagged, 0)
        self.assertEqual(rejected, 0)

    def test_submission_format_unchanged_and_reports_zero_joint_flags(self):
        with tempfile.TemporaryDirectory() as tmp:
            src_in, tpl, out = Path(tmp) / "t.csv", Path(tmp) / "s.csv", Path(tmp) / "o.csv"
            pd.read_csv(config.TEST_CSV, nrows=2000).to_csv(src_in, index=False)
            pd.read_csv(config.SAMPLE_SUBMISSION_CSV, nrows=2000).to_csv(tpl, index=False)
            summary = predict.generate_submission(src_in, out, tpl, chunksize=700)
            self.assertEqual(summary["rows_with_feature_sum_outside_training_range"], 0)
            self.assertTrue(summary["template_check"]["passed"])
            self.assertEqual(list(pd.read_csv(out).columns), [config.ID_COL, config.TARGET])

    @unittest.skipUnless(SUBMISSION.exists(), "submission.csv not present")
    def test_existing_submission_values_are_reproduced_exactly(self):
        head_test = pd.read_csv(config.TEST_CSV, nrows=5000)
        result, _ = predict.predict_batch(head_test)
        saved = pd.read_csv(SUBMISSION, nrows=5000)
        self.assertTrue((result[config.ID_COL].to_numpy() == saved[config.ID_COL].to_numpy()).all())
        np.testing.assert_allclose(result[config.TARGET], saved[config.TARGET], rtol=0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
