"""Tests for the presentation-only percentage display.

Rules being protected:
* predictions are DISPLAYED as percentages with one decimal (0.868 -> 86.8%, 0.5113 -> 51.1%);
* downloadable CSVs keep the raw decimal predictions (no '%', no scaling, no clipping);
* MAE / RMSE / R² keep their original units;
* the extrapolation warnings still appear next to the percentage result;
* the prediction, training and export code was not changed to do any of this.
"""
import io
import json
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from src import charts, config, formatting, predict
from src.formatting import DATASET_SCORE_NOTE, PERCENT_COLUMN_LABEL, format_percent, format_predictions_for_display
from src.train import regression_metrics
from tests.test_dashboard import _Upload, _metric_calls, _warning_texts, run_app

FIXTURE = config.ROOT / "tests" / "fixtures" / "baseline_train_rows.csv"


def uniform_state(value):
    return {f"in_{f}": value for f in config.FEATURES}


def sample_upload():
    rows = pd.DataFrame([{**{config.ID_COL: i}, **{f: v for f in config.FEATURES}} for i, v in [(1, 5), (2, 10), (3, 15)]])
    return _Upload(rows.to_csv(index=False).encode())


def texts(st, method):
    return [str(c.args[0]) for c in getattr(st, method).call_args_list if c.args]


class TestFormatPercent(unittest.TestCase):
    def test_examples_from_the_requirements(self):
        self.assertEqual(format_percent(0.868), "86.8%")
        self.assertEqual(format_percent(0.5113), "51.1%")

    def test_one_decimal_place_and_half_up_rounding(self):
        cases = {0.0: "0.0%", 1.0: "100.0%", 0.285: "28.5%", 0.725: "72.5%", 0.8675: "86.8%",
                 0.8665: "86.7%", 0.30240874454605127: "30.2%", 0.7599705159161959: "76.0%", 0.5: "50.0%"}
        for value, expected in cases.items():
            with self.subTest(value=value):
                self.assertEqual(format_percent(value), expected)

    def test_values_outside_zero_one_are_not_clipped(self):
        self.assertEqual(format_percent(1.075959), "107.6%")
        self.assertEqual(format_percent(1.640601), "164.1%")
        self.assertEqual(format_percent(-0.123), "-12.3%")

    def test_tiny_negative_is_not_shown_as_negative_zero(self):
        self.assertEqual(format_percent(-0.0004), "0.0%")

    def test_numpy_and_pandas_scalars(self):
        self.assertEqual(format_percent(np.float64(0.868)), "86.8%")
        self.assertEqual(format_percent(pd.Series([0.5113]).iloc[0]), "51.1%")

    def test_bad_input_gives_na(self):
        for bad in (np.nan, np.inf, None, "abc"):
            with self.subTest(bad=bad):
                self.assertEqual(format_percent(bad), "n/a")

    def test_decimals_argument(self):
        self.assertEqual(format_percent(0.5113, decimals=0), "51%")
        self.assertEqual(format_percent(0.5113, decimals=2), "51.13%")


class TestDisplayTable(unittest.TestCase):
    def test_display_frame_uses_percentages_and_leaves_input_untouched(self):
        raw = pd.DataFrame({config.ID_COL: [1, 2], config.TARGET: [0.868, 0.5113], "extrapolation_warning": [False, True]})
        before = raw.copy(deep=True)
        shown = format_predictions_for_display(raw)
        self.assertEqual(shown[PERCENT_COLUMN_LABEL].tolist(), ["86.8%", "51.1%"])
        self.assertEqual(shown["extrapolation_warning"].tolist(), [False, True])
        self.assertEqual(shown[config.ID_COL].tolist(), [1, 2])
        pd.testing.assert_frame_equal(raw, before)                         # raw decimals untouched
        self.assertEqual(raw[config.TARGET].tolist(), [0.868, 0.5113])

    def test_row_limit(self):
        raw = pd.DataFrame({config.ID_COL: range(300), config.TARGET: 0.5})
        self.assertEqual(len(format_predictions_for_display(raw, max_rows=100)), 100)
        self.assertEqual(len(format_predictions_for_display(raw, max_rows=None)), 300)


class TestSinglePredictionPage(unittest.TestCase):
    def test_in_domain_result_card_is_a_percentage_with_dataset_score_label(self):
        st = run_app("Flood Probability Prediction", click_predict=True, state=uniform_state(5))
        metrics = _metric_calls(st)
        card = [m for m in metrics if m[0].startswith("Predicted FloodProbability")]
        self.assertEqual(card, [("Predicted FloodProbability score (model estimate)", "51.1%")])
        self.assertTrue(any(DATASET_SCORE_NOTE in c for c in texts(st, "caption")))
        self.assertIn("not a verified real-world flood probability", DATASET_SCORE_NOTE)
        self.assertFalse(any("0.5113" in str(m[1]) for m in metrics))      # no raw decimal on the card
        self.assertFalse(any("Extrapolation warning" in w for w in _warning_texts(st)))

    def test_requirement_example_086_8_percent(self):
        fake = predict.SinglePrediction(value=0.868, feature_sum=120.0, sum_range=(63, 144), outside_joint_range=False)
        with mock.patch.object(predict, "predict_single_with_checks", return_value=fake):
            st = run_app("Flood Probability Prediction", click_predict=True)
        self.assertIn(("Predicted FloodProbability score (model estimate)", "86.8%"), _metric_calls(st))

    def test_extrapolation_warning_is_preserved_next_to_percentage_result(self):
        st = run_app("Flood Probability Prediction", click_predict=True, state=uniform_state(10))
        warnings = _warning_texts(st)
        warning = [w for w in warnings if "Extrapolation warning" in w]
        self.assertEqual(len(warning), 1, warnings)
        for fragment in ("200", "63-144", "28.5%-72.5%", "above 100%", "not be read as a probability"):
            self.assertIn(fragment, warning[0])
        self.assertIn(
        ("Predicted FloodProbability score (model estimate)", "100.0%"),
        _metric_calls(st),
        )
        self.assertTrue(any(DATASET_SCORE_NOTE in c for c in texts(st, "caption")))
        self.assertFalse(any("1.0760" in str(m[1]) for m in _metric_calls(st)))

    def test_extreme_input_is_displayed_capped(self):
        st = run_app(
            "Flood Probability Prediction",
            click_predict=True,
            state=uniform_state(15),
        )

        warnings = _warning_texts(st)
        self.assertTrue(
            any("Extrapolation warning" in w for w in warnings),
            warnings,
        )

        metrics = _metric_calls(st)
        displayed = [
            m for m in metrics
            if m[0].startswith("Predicted FloodProbability")
        ]

        self.assertEqual(len(displayed), 1, metrics)
        self.assertEqual(displayed[0][1], "100.0%")


class TestBatchPage(unittest.TestCase):
    def setUp(self):
        self.st = run_app("Batch Prediction", upload=sample_upload())
        downloads = [c for c in self.st.download_button.call_args_list if c.args[2] == "flood_predictions.csv"]
        self.assertEqual(len(downloads), 1)
        self.csv_bytes = downloads[0].args[1]

    def table_shown(self):
        frames = [c.args[0] for c in self.st.dataframe.call_args_list
                  if c.args and isinstance(c.args[0], pd.DataFrame) and PERCENT_COLUMN_LABEL in c.args[0].columns]
        self.assertEqual(len(frames), 1)
        return frames[0]

    def test_dashboard_table_shows_percentages(self):
        shown = self.table_shown()
        self.assertEqual(shown[PERCENT_COLUMN_LABEL].tolist(), ["51.1%", "107.6%", "164.1%"])
        self.assertNotIn(config.TARGET, shown.columns)

    def test_download_csv_keeps_raw_decimals_and_original_format(self):
        text = self.csv_bytes.decode("utf-8")
        self.assertNotIn("%", text)
        self.assertEqual(text.splitlines()[0], f"{config.ID_COL},{config.TARGET},extrapolation_warning")
        out = pd.read_csv(io.BytesIO(self.csv_bytes))
        expected, _ = predict.predict_batch(pd.read_csv(io.StringIO(sample_upload().getvalue().decode())), include_flags=True)
        np.testing.assert_allclose(out[config.TARGET], expected[config.TARGET], rtol=0, atol=1e-12)
        self.assertEqual(out[config.TARGET].round(4).tolist(), [0.5113, 1.0760, 1.6406])   # not x100, not clipped
        self.assertTrue(pd.api.types.is_float_dtype(out[config.TARGET]))

    def test_extrapolation_flags_counts_and_warnings_are_preserved(self):
        out = pd.read_csv(io.BytesIO(self.csv_bytes))
        self.assertEqual(out["extrapolation_warning"].tolist(), [False, True, True])
        self.assertEqual(dict(_metric_calls(self.st))["Feature sum outside training range"], "2")
        self.assertTrue(any("feature sum" in w for w in _warning_texts(self.st)))
        caption = " ".join(texts(self.st, "caption"))
        self.assertIn("raw decimal", caption)
        self.assertIn("not a verified real-world flood probability", caption)


class TestRealRowsAndRawCsvValues(unittest.TestCase):
    def test_real_training_rows_download_values_equal_recorded_raw_predictions(self):
        fx = pd.read_csv(FIXTURE)
        upload = _Upload(fx[[config.ID_COL] + config.FEATURES].to_csv(index=False).encode())
        st = run_app("Batch Prediction", upload=upload)
        download = [c for c in st.download_button.call_args_list if c.args[2] == "flood_predictions.csv"][0]
        out = pd.read_csv(io.BytesIO(download.args[1]))
        np.testing.assert_allclose(out[config.TARGET], fx["expected_prediction"], rtol=0, atol=1e-12)
        self.assertFalse(out["extrapolation_warning"].any())

    def test_command_line_submission_format_is_unchanged(self):
        # generate_submission does not import or use the formatting helpers
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            src_in, out = Path(tmp) / "in.csv", Path(tmp) / "out.csv"
            fx = pd.read_csv(FIXTURE)
            fx[[config.ID_COL] + config.FEATURES].head(50).to_csv(src_in, index=False)
            predict.generate_submission(src_in, out)
            text = out.read_text()
            self.assertNotIn("%", text)
            self.assertEqual(text.splitlines()[0], f"{config.ID_COL},{config.TARGET}")
            np.testing.assert_allclose(pd.read_csv(out)[config.TARGET], fx["expected_prediction"].head(50), rtol=0, atol=1e-12)


class TestMetricsKeepOriginalUnits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metrics = json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))
        cls.best = cls.metrics["models"][cls.metrics["best_model"]]

    def test_metric_values_are_recomputed_from_raw_validation_predictions(self):
        preds = pd.read_csv(config.VALIDATION_PREDICTIONS_PATH)
        recomputed = regression_metrics(preds["actual"], preds["predicted"])
        for key in ("MAE", "RMSE", "R2"):
            self.assertAlmostEqual(recomputed[key], self.best[key], places=9)
        self.assertLess(self.best["MAE"], 0.1)       # score units (0.0158), not percent (1.58)

    def test_home_page_labels_and_values_are_original_units(self):
        st = run_app("Home")
        metrics = dict(_metric_calls(st))
        self.assertEqual(metrics["MAE (score units)"], f"{self.best['MAE']:.5f}")
        self.assertEqual(metrics["RMSE (score units)"], f"{self.best['RMSE']:.5f}")
        self.assertEqual(metrics["R² (unitless)"], f"{self.best['R2']:.4f}")
        for label in ("MAE (score units)", "RMSE (score units)", "R² (unitless)"):
            self.assertNotIn("%", metrics[label])

    def test_model_performance_table_is_in_original_units(self):
        st = run_app("Model Performance")
        tables = [c.args[0] for c in st.dataframe.call_args_list if c.args and isinstance(c.args[0], pd.DataFrame)
                  and "MAE (score units)" in c.args[0].columns]
        self.assertEqual(len(tables), 1)
        table = tables[0]
        self.assertEqual(table.loc[self.metrics["best_model"], "MAE (score units)"], f"{self.best['MAE']:.5f}")
        self.assertFalse(table["MAE (score units)"].str.contains("%").any())
        self.assertTrue(any("original units" in c for c in texts(st, "caption")))


class TestPresentationOnly(unittest.TestCase):
    def test_prediction_training_and_export_code_do_not_use_the_formatting_helpers(self):
        for name in ("predict.py", "train.py", "evaluate.py", "data_processing.py"):
            source = (config.ROOT / "src" / name).read_text(encoding="utf-8")
            self.assertNotIn("formatting", source, name)
            self.assertNotIn("format_percent", source, name)

    def test_raw_model_output_is_unchanged_and_unclipped(self):
        self.assertAlmostEqual(predict.predict_single({f: 5 for f in config.FEATURES}), 0.5113, places=4)
        self.assertAlmostEqual(predict.predict_single({f: 10 for f in config.FEATURES}), 1.0760, places=4)

    def test_contribution_chart_labels_are_percentage_points_but_data_unchanged(self):
        contributions = pd.Series(np.linspace(-0.02, 0.02, 20), index=config.FEATURES)
        fig = charts.plot_feature_contributions(contributions)
        ax = fig.axes[0]
        self.assertIn("percentage points", ax.get_xlabel())
        self.assertEqual(ax.xaxis.get_major_formatter()(0.01, 0), "+1.0")
        self.assertEqual(ax.xaxis.get_major_formatter()(-0.025, 0), "-2.5")
        plotted = sorted(p.get_width() for p in ax.patches)
        np.testing.assert_allclose(plotted, sorted(contributions.values))   # bars still in score units


if __name__ == "__main__":
    unittest.main()
