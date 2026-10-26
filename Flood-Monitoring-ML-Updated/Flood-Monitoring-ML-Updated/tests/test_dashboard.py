"""Dashboard tests.

1. Always runs: every page of app.py is executed against a stand-in `streamlit` object. This
   verifies the Python logic, data loading and chart construction, NOT real Streamlit rendering.
2. Runs only if Streamlit is installed: Streamlit's official AppTest starts the real app.
"""
import importlib.util
import io
import runpy
import sys
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock

import pandas as pd

from src import config

APP = str(config.ROOT / "app.py")
PAGES = ["Home", "Flood Probability Prediction", "Data Explorer", "Model Performance",
         "Batch Prediction", "About the Project"]


class _Upload:
    def __init__(self, data: bytes):
        self._data = data

    def getvalue(self):
        return self._data


def make_fake_streamlit(page, click_predict=False, upload=None, state=None):
    st = MagicMock(name="streamlit")
    state = dict(state or {})
    st.session_state = state
    passthrough = lambda *a, **k: a[0] if (a and callable(a[0]) and not k) else (lambda f: f)
    st.cache_data, st.cache_resource = passthrough, passthrough
    st.radio = lambda label, options, key=None, **k: page
    # Columns and tabs delegate to the same stand-in, so c[0].metric(...) is recorded like st.metric(...).
    st.columns = lambda spec, **k: [st for _ in range(spec if isinstance(spec, int) else len(spec))]
    st.tabs = lambda labels: [st for _ in labels]

    def slider(label, *args, **kwargs):
        if "key" in kwargs:
            return state[kwargs["key"]]
        return args[2] if len(args) > 2 else args[0]
    st.slider = slider
    st.selectbox = lambda label, options, *a, **k: list(options)[0]
    st.button = lambda label, *a, **k: bool(click_predict and label == "Predict")
    st.checkbox = lambda *a, **k: False
    st.file_uploader = lambda *a, **k: upload
    module = ModuleType("streamlit")
    module.__dict__.update({k: getattr(st, k) for k in
                            ["cache_data", "cache_resource", "radio", "columns", "tabs", "slider", "selectbox",
                             "button", "checkbox", "file_uploader", "session_state"]})
    module.__getattr__ = lambda name: getattr(st, name)  # everything else is a harmless mock
    return module, st


def run_app(page, **kwargs):
    module, st = make_fake_streamlit(page, **kwargs)
    saved = sys.modules.get("streamlit")
    sys.modules["streamlit"] = module
    try:
        runpy.run_path(APP, run_name="__main__")
    finally:
        if saved is None:
            sys.modules.pop("streamlit", None)
        else:
            sys.modules["streamlit"] = saved
    return st


class TestDashboardLogicWithStandInStreamlit(unittest.TestCase):
    def test_app_source_compiles(self):
        compile(Path(APP).read_text(encoding="utf-8"), APP, "exec")

    def test_every_page_runs_without_error(self):
        for page in PAGES:
            with self.subTest(page=page):
                run_app(page)

    def test_predict_button_produces_a_prediction(self):
        st = run_app("Flood Probability Prediction", click_predict=True)
        shown = [c.args for c in st.metric.call_args_list]
        self.assertTrue(any("Predicted FloodProbability" in a[0] and 20 < float(a[1].rstrip("%")) < 80
                            for a in shown), shown)

    def test_batch_page_with_valid_and_invalid_upload(self):
        rows = pd.DataFrame([{**{config.ID_COL: i}, **{f: 5 for f in config.FEATURES}} for i in range(5)])
        rows.loc[2, "Landslides"] = -3
        st = run_app("Batch Prediction", upload=_Upload(rows.to_csv(index=False).encode()))
        labels = {c.args[0]: c.args[1] for c in st.metric.call_args_list}
        self.assertEqual(labels["Rows predicted"], "4")
        self.assertEqual(labels["Rows rejected"], "1")

    def test_batch_page_missing_columns_shows_error(self):
        bad = pd.DataFrame({config.ID_COL: [1], "Landslides": [3]})
        st = run_app("Batch Prediction", upload=_Upload(bad.to_csv(index=False).encode()))
        self.assertTrue(st.error.called)



def _warning_texts(st):
    return [str(c.args[0]) for c in st.warning.call_args_list if c.args]


def _metric_calls(st):
    return [(c.args[0], c.args[1]) for c in st.metric.call_args_list]


class TestExtrapolationWarningInDashboard(unittest.TestCase):
    """Warn-and-show on the single-prediction page; flags and counts on the batch page."""

    def uniform_state(self, value):
        return {f"in_{f}": value for f in config.FEATURES}

    def test_out_of_domain_inputs_show_warning_and_display_capped_value(self):
        st = run_app(
            "Flood Probability Prediction",
            click_predict=True,
            state=self.uniform_state(10),
        )

        warnings = _warning_texts(st)
        self.assertTrue(
            any(
                "Extrapolation warning" in w
                and "200" in w
                and "63-144" in w
                for w in warnings
            ),
            warnings,
        )

        metrics = _metric_calls(st)
        displayed = [
            m for m in metrics
            if m[0].startswith("Predicted FloodProbability")
        ]

        self.assertEqual(len(displayed), 1, metrics)
        self.assertEqual(displayed[0][1], "100.0%")

    def test_in_domain_inputs_show_normal_result_and_no_extrapolation_warning(self):
        st = run_app("Flood Probability Prediction", click_predict=True, state=self.uniform_state(5))
        self.assertFalse(any("Extrapolation warning" in w for w in _warning_texts(st)))
        metrics = _metric_calls(st)
        normal = [m for m in metrics if m[0].startswith("Predicted FloodProbability")]
        self.assertEqual(normal[0][1], "51.1%")

    def test_default_example_presets_are_inside_the_training_domain(self):
        st = run_app("Flood Probability Prediction", click_predict=True)      # default = median preset
        self.assertFalse(any("Extrapolation warning" in w for w in _warning_texts(st)))

    def test_batch_page_counts_out_of_range_rows_and_download_contains_flags(self):
        rows = pd.DataFrame([{**{config.ID_COL: i}, **{f: v for f in config.FEATURES}}
                             for i, v in [(1, 5), (2, 10), (3, 15)]])
        st = run_app("Batch Prediction", upload=_Upload(rows.to_csv(index=False).encode()))
        labels = dict(_metric_calls(st))
        self.assertEqual(labels["Rows predicted"], "3")
        self.assertEqual(labels["Feature sum outside training range"], "2")
        self.assertEqual(labels["Single feature outside training range"], "0")
        self.assertTrue(any("feature sum" in w for w in _warning_texts(st)))
        downloads = [c for c in st.download_button.call_args_list if c.args[2] == "flood_predictions.csv"]
        self.assertEqual(len(downloads), 1)
        out = pd.read_csv(io.BytesIO(downloads[0].args[1]))
        self.assertEqual(list(out.columns), [config.ID_COL, config.TARGET, "extrapolation_warning"])
        self.assertEqual(out["extrapolation_warning"].tolist(), [False, True, True])
        self.assertAlmostEqual(out[config.TARGET].iloc[2], 1.6406, places=4)   # not clipped

    def test_batch_page_in_range_rows_have_no_flags(self):
        fx = pd.read_csv(config.ROOT / "tests" / "fixtures" / "baseline_train_rows.csv")
        upload = _Upload(fx[[config.ID_COL] + config.FEATURES].to_csv(index=False).encode())
        st = run_app("Batch Prediction", upload=upload)
        self.assertEqual(dict(_metric_calls(st))["Feature sum outside training range"], "0")
        downloads = [c for c in st.download_button.call_args_list if c.args[2] == "flood_predictions.csv"]
        out = pd.read_csv(io.BytesIO(downloads[0].args[1]))
        self.assertFalse(out["extrapolation_warning"].any())


@unittest.skipUnless(importlib.util.find_spec("streamlit") is not None, "streamlit not installed")
class TestRealStreamlitStartup(unittest.TestCase):
    def test_app_starts_and_all_pages_render(self):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(APP, default_timeout=120).run()
        self.assertFalse(at.exception, at.exception)
        for page in PAGES:
            at.sidebar.radio(key="page").set_value(page).run()
            self.assertFalse(at.exception, f"{page}: {at.exception}")


if __name__ == "__main__":
    unittest.main()
