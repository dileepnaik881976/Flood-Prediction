"""Flood Monitoring and Prediction System Using Machine Learning - Streamlit dashboard.

Run from the project root:   streamlit run app.py
The dashboard only LOADS the trained model and the precomputed reports; it never retrains.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure

from src import charts, config
from src.feature_info import FEATURE_DESCRIPTIONS
from src.formatting import (
    DATASET_SCORE_NOTE,
    format_percent,
    format_prediction_score,
    format_predictions_for_display,
)
from src.predict import (FLAG_COLUMN, PredictionInputError, linear_contributions, load_model, load_schema,
                         predict_batch, predict_single_with_checks)

DISCLAIMER = ("Experimental student project. Outputs are model estimates of a dataset score. They are NOT "
              "official flood forecasts or emergency warnings. Do not use them for safety decisions.")
PAGES = ["Home", "Flood Probability Prediction", "Data Explorer", "Model Performance",
         "Batch Prediction", "About the Project"]
DATA_SUMMARY_PATH = config.REPORTS_DIR / "data_summary.json"
TRAIN_SAMPLE_PATH = config.REPORTS_DIR / "train_sample.csv"

st.set_page_config(page_title="Flood Probability Dashboard", page_icon="🌊", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
[data-testid="stMetric"] {background: #F3F6FA; border-radius: 8px; padding: 12px 16px;}
h1, h2, h3 {letter-spacing: -0.01em;}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ cached loaders
@st.cache_resource(show_spinner=False)
def get_artifacts():
    return load_model(), load_schema()


@st.cache_data(show_spinner=False)
def read_json(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def run_batch(file_bytes: bytes, strict_range: bool):
    df = pd.read_csv(io.BytesIO(file_bytes))
    result, report = predict_batch(df, strict_range=strict_range, include_flags=True)
    return df.shape, result, report, result.to_csv(index=False).encode("utf-8")


def load_everything():
    needed = [config.MODEL_PATH, config.SCHEMA_PATH, config.METRICS_PATH, DATA_SUMMARY_PATH, TRAIN_SAMPLE_PATH,
              config.VALIDATION_PREDICTIONS_PATH, config.FEATURE_IMPORTANCE_PATH]
    missing = [p.name for p in needed if not p.exists()]
    if missing:
        st.error(f"Missing project files: {missing}. From the project root run "
                 "`python -m src.train` then `python -m src.evaluate`.")
        st.stop()
    model, schema = get_artifacts()
    return (model, schema, read_json(str(config.METRICS_PATH)), read_json(str(DATA_SUMMARY_PATH)),
            read_csv(str(TRAIN_SAMPLE_PATH)))


model, schema, metrics, summary, sample = load_everything()
best = metrics["best_model"]
best_metrics = metrics["models"][best]


@st.cache_resource(show_spinner=False)
def cached_figure(name: str) -> Figure:
    if name == "target":
        return charts.plot_target_distribution(sample[config.TARGET])
    if name == "features":
        return charts.plot_feature_distributions(sample)
    if name == "corr":
        c = summary["correlation"]
        return charts.plot_correlation_heatmap(pd.DataFrame(c["matrix"], index=c["columns"], columns=c["columns"]))
    if name == "compare":
        return charts.plot_metric_comparison(metrics["models"])
    preds = read_csv(str(config.VALIDATION_PREDICTIONS_PATH))
    if name == "avp":
        return charts.plot_actual_vs_predicted(preds["actual"], preds["predicted"])
    if name == "resid":
        return charts.plot_residuals(preds["actual"], preds["predicted"])
    if name == "importance":
        return charts.plot_feature_importance(read_csv(str(config.FEATURE_IMPORTANCE_PATH)))
    raise KeyError(name)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🌊 Flood Probability")
    page = st.radio("Navigate", PAGES, key="page")
    st.divider()
    st.caption(f"Model: **{best}**")
    st.caption(f"Validation R²: **{best_metrics['R2']:.3f}**")
    st.caption(DISCLAIMER)


# ------------------------------------------------------------------ pages
def page_home():
    st.title("Flood Monitoring and Prediction System Using Machine Learning")
    st.markdown("A dashboard that estimates a dataset-based **FloodProbability score** from 20 numeric "
                "risk-factor scores, explores the data, evaluates models, and predicts from uploaded CSV files.")
    st.warning(DISCLAIMER)

    st.subheader("How the model works")
    st.markdown(
        "- Each row describes a region with **20 integer scores** (for example `MonsoonIntensity`, `Urbanization`).\n"
        "- The target `FloodProbability` is a **continuous number**, so this is a *regression* problem.\n"
        "- Several models were trained on 80% of `train.csv` and compared on the remaining 20%. "
        "The best validation score was kept and saved.\n"
        "- This version has **no live sensor data, timestamps, or verified flood events**. "
        "It is not real-time flood detection.")

    st.subheader("Dataset statistics")
    c = st.columns(4)
    c[0].metric("train.csv rows", f"{summary['train']['rows']:,}")
    c[1].metric("test.csv rows", f"{summary['test']['rows']:,}")
    c[2].metric("flood.csv rows", f"{summary['flood']['rows']:,}")
    c[3].metric("Input features", len(config.FEATURES))
    c = st.columns(3)
    c[0].metric("Missing values (train)", f"{summary['train']['missing_values']:,}")
    c[1].metric("Duplicate feature rows (train)", f"{summary['train']['duplicate_feature_rows']:,}")
    c[2].metric("Target range (score units, 0-1)", f"{summary['target_stats']['min']:.3f} - {summary['target_stats']['max']:.3f}")

    st.subheader("Selected model and validation metrics")
    st.markdown(f"Selected model: **{best}** (lowest validation RMSE among the compared models).")
    c = st.columns(3)
    c[0].metric("MAE (score units)", f"{best_metrics['MAE']:.5f}")
    c[1].metric("RMSE (score units)", f"{best_metrics['RMSE']:.5f}")
    c[2].metric("R² (unitless)", f"{best_metrics['R2']:.4f}")
    st.caption("Error metrics are in the original units of the FloodProbability score (a 0-1 dataset score). "
               "They are not percentages; predictions elsewhere in the dashboard are displayed as percentages.")
    st.caption(f"Validation split: {metrics['split']['validation_rows']:,} rows, seed {metrics['split']['seed']}.")


def _set_inputs(values: dict):
    for f, v in values.items():
        st.session_state[f"in_{f}"] = int(v)


def page_predict():
    st.title("Flood Probability Prediction")
    st.warning(DISCLAIMER)
    stats, ranges = summary["feature_stats"], schema["feature_ranges"]
    presets = {
        "Typical (median of each feature)": {f: round(stats[f]["50%"]) for f in config.FEATURES},
        "Lower inputs (25th percentile)": {f: round(stats[f]["25%"]) for f in config.FEATURES},
        "Higher inputs (75th percentile)": {f: round(stats[f]["75%"]) for f in config.FEATURES},
    }
    for f in config.FEATURES:
        st.session_state.setdefault(f"in_{f}", presets["Typical (median of each feature)"][f])

    st.markdown("Set the 20 feature scores (allowed range = range observed in the training data), "
                "then press **Predict**. Example presets are illustrative inputs, not real scenarios.")
    cols = st.columns(len(presets) + 1)
    for col, (label, vals) in zip(cols, presets.items()):
        col.button(label, on_click=_set_inputs, args=(vals,), key=f"preset_{label}")
    cols[-1].button("Reset", on_click=_set_inputs, args=(presets["Typical (median of each feature)"],), key="reset")

    left, right = st.columns(2)
    for i, f in enumerate(config.FEATURES):
        with (left if i % 2 == 0 else right):
            st.slider(f, min_value=ranges[f]["min"], max_value=ranges[f]["max"], key=f"in_{f}",
                      help=FEATURE_DESCRIPTIONS[f])
    values = {f: st.session_state[f"in_{f}"] for f in config.FEATURES}
    sum_low, sum_high = schema["feature_sum_range"]
    total = sum(values.values())
    st.caption(f"Current feature sum: **{total}**. Training data range: {sum_low}-{sum_high}"
               + ("." if sum_low <= total <= sum_high else " (outside: predictions will be extrapolations)."))

    if st.button("Predict", type="primary", key="predict"):
        try:
            st.session_state["last_prediction"] = (dict(values), predict_single_with_checks(values))
        except PredictionInputError as exc:
            st.session_state.pop("last_prediction", None)
            st.error(f"Input validation failed: {exc}")

    if "last_prediction" in st.session_state:
        used, result = st.session_state["last_prediction"]
        if used != values:
            st.info("Inputs changed since the last prediction. Press Predict to update.")
        t_min, t_max = summary["target_stats"]["min"], summary["target_stats"]["max"]
        if result.outside_joint_range:

            st.warning(
                f"**Extrapolation warning.** These inputs add up to {result.feature_sum:.0f}, outside the "
                f"{result.sum_range[0]}-{result.sum_range[1]} range of feature sums seen in training. "
                "The model has never seen such a combination, so its raw output is a straight-line "
                "extrapolation. The raw output can be above 100%, but the displayed score is bounded "
                "to 1.0%-100.0%. The training dataset's FloodProbability scores ranged from "
                "28.5%-72.5%; the displayed score should not be read as a probability "
                "of real-world flooding."
            )
            st.metric(
                "Predicted FloodProbability score (model estimate)",
                format_prediction_score(result.value)
            )
            st.caption(DATASET_SCORE_NOTE)
            st.caption(
                "The displayed score is bounded for presentation only. The underlying model output and downloaded "
                "CSV values remain unchanged. This dataset score is not necessarily a calibrated real-world "
                "flood probability."
            )

        else:
            st.metric("Predicted FloodProbability score (model estimate)", format_percent(result.value))
            pct = float((sample[config.TARGET] < result.value).mean() * 100)
            st.caption(f"{DATASET_SCORE_NOTE} For context: this score is higher than {pct:.0f}% of the "
                       "FloodProbability values in the training sample. It is not necessarily a calibrated "
                       "real-world probability, and no risk categories are assigned.")
        means = {f: stats[f]["mean"] for f in config.FEATURES}
        contributions = linear_contributions(used, means)
        if contributions is not None:
            with st.expander("What moved this prediction? (exact for the linear model)"):
                st.pyplot(charts.plot_feature_contributions(contributions))
                st.caption("Each bar = coefficient × (your input − dataset average), shown in percentage points of "
                           "the displayed score. Associations in the data, not proof of cause and effect.")


def page_explorer():
    st.title("Data Explorer")
    st.caption("Statistics come from the full train.csv. Charts and row browsing use a stored random sample of "
               f"{len(sample):,} rows, so the browser never loads the full million-row file.")
    tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Distributions", "Correlation", "Browse rows"])

    with tab1:
        c = st.columns(3)
        for col, key in zip(c, ["train", "test", "flood"]):
            col.metric(f"{key}.csv", f"{summary[key]['rows']:,} rows × {summary[key]['columns']} cols")
        quality = pd.DataFrame({k: {"Rows": summary[k]["rows"], "Missing values": summary[k]["missing_values"],
                                    "Duplicate feature rows": summary[k]["duplicate_feature_rows"]}
                                for k in ["train", "test", "flood"]}).T
        st.dataframe(quality)
        table = pd.DataFrame(summary["feature_stats"]).T.round(3)
        table.insert(0, "Description (generic, not official)", [FEATURE_DESCRIPTIONS[f] for f in table.index])
        st.markdown("**Feature descriptive statistics (train.csv)**")
        st.dataframe(table)
        st.caption("The datasets include no data dictionary; descriptions are generic readings of the column names.")

    with tab2:
        st.pyplot(cached_figure("target"))
        st.pyplot(cached_figure("features"))
        feature = st.selectbox("Inspect one feature", config.FEATURES)
        counts = sample[feature].value_counts().sort_index()
        fig = Figure(figsize=(7, 3.5), layout="constrained")
        ax = fig.subplots()
        ax.bar(counts.index, counts.values, color=charts.BLUE)
        ax.set_xlabel(f"{feature} (integer score)")
        ax.set_ylabel("Number of rows in sample")
        ax.set_title(f"Distribution of {feature}")
        st.pyplot(fig)

    with tab3:
        st.pyplot(cached_figure("corr"))
        c = summary["correlation"]
        corr = pd.DataFrame(c["matrix"], index=c["columns"], columns=c["columns"])
        st.markdown("**Correlation with FloodProbability**")
        st.dataframe(corr[config.TARGET].drop(config.TARGET).sort_values(ascending=False).rename("correlation").round(4))
        st.caption("Correlation measures linear association only and does not prove causation.")

    with tab4:
        c1, c2 = st.columns(2)
        feature = c1.selectbox("Filter by feature", ["(none)"] + config.FEATURES)
        n_rows = c2.slider("Rows to display", 10, 1000, 100, step=10)
        filtered = sample
        if feature != "(none)":
            lo, hi = int(sample[feature].min()), int(sample[feature].max())
            chosen = st.slider(f"{feature} range", lo, hi, (lo, hi))
            filtered = filtered[filtered[feature].between(*chosen)]
        t_lo, t_hi = float(sample[config.TARGET].min()), float(sample[config.TARGET].max())
        t_range = st.slider("FloodProbability range", t_lo, t_hi, (t_lo, t_hi), step=0.005)
        filtered = filtered[filtered[config.TARGET].between(*t_range)]
        st.write(f"{len(filtered):,} of {len(sample):,} sample rows match. Showing up to {n_rows}.")
        st.dataframe(filtered.head(n_rows))


def page_performance():
    st.title("Model Performance")
    st.caption(f"All numbers come from the saved training run: {metrics['split']['train_rows']:,} training rows, "
               f"{metrics['split']['validation_rows']:,} validation rows, seed {metrics['split']['seed']}.")
    table = pd.DataFrame(metrics["models"]).T[["MAE", "RMSE", "R2", "train_rows_used", "fit_seconds"]]
    table.columns = ["MAE (score units)", "RMSE (score units)", "R² (unitless)", "Training rows used", "Fit time (s)"]
    st.caption("MAE, RMSE and R² are computed on the original 0-1 FloodProbability score and are shown in those "
               "original units, not as percentages.")
    shown = table.copy()
    for col, fmt in {"MAE (score units)": "{:.5f}", "RMSE (score units)": "{:.5f}", "R² (unitless)": "{:.4f}",
                     "Training rows used": "{:,.0f}", "Fit time (s)": "{:.1f}"}.items():
        shown[col] = shown[col].map(fmt.format)
    st.dataframe(shown)
    st.success(f"Selected: {best} (lowest validation RMSE excluding the mean baseline).")
    st.caption("Random Forest and Extra Trees were fitted on a row subsample (see 'Training rows used'), so the "
               "comparison with models fitted on all training rows is not like for like.")
    st.pyplot(cached_figure("compare"))
    c1, c2 = st.columns([1, 2])
    with c1:
        st.pyplot(cached_figure("avp"))
    with c2:
        st.pyplot(cached_figure("resid"))
    res = summary["validation_residuals"]
    sum_low, sum_high = schema["feature_sum_range"]
    st.markdown(f"Residuals are centred near zero (mean {res['mean']:.5f}, std {res['std']:.4f}, in score units). "
                f"{res['within_0.02']*100:.1f}% of validation predictions are within ±0.02 of the actual value and "
                f"{res['within_0.05']*100:.1f}% within ±0.05.")
    st.pyplot(cached_figure("importance"))
    ext = metrics.get("external_flood_csv")
    if ext:
        with st.expander("External check on flood.csv (never used for training or model selection)"):
            st.write(f"MAE {ext['MAE']:.5f} (score units) | RMSE {ext['RMSE']:.5f} (score units) | R² {ext['R2']:.4f}")
            st.caption("flood.csv's target is an exact formula (0.005 × sum of the 20 features), so this check "
                       "shows compatibility with a related dataset, not real-world accuracy.")
    st.subheader("Limitations in plain language")
    st.markdown(
        f"- **Score, not a forecast.** The model predicts this dataset's `FloodProbability` number. It is not "
        "calibrated against verified flood events.\n"
        f"- **Not perfect.** R² ≈ {best_metrics['R2']:.2f}: about {100*(1-best_metrics['R2']):.0f}% of the variation "
        "in the target is not explained by the 20 features.\n"
        "- **Anonymous scores.** Features are integer scores with no location, date or units, and the data has no "
        "live readings.\n"
        "- **A simple model won.** A linear model beat the tree models here, which suggests the target is "
        "mostly an additive combination of the features. Tree models were also given less training data.\n"
        f"- **Extrapolation.** Inputs beyond the training data get unreliable predictions. This includes "
        f"combinations where the 20 features add up to less than {sum_low} or more than {sum_high} (the range seen in "
        "training): a linear model keeps rising and can output values above 1, for example when every feature is 10.\n"
        "- **Correlation is not causation.** Importance and coefficients describe the dataset only.")


def page_batch():
    st.title("Batch Prediction")
    st.warning(DISCLAIMER)
    st.markdown("Upload a CSV with these **20 numeric columns** (any order). An `id` column is kept in the output. "
                "Other columns are ignored.")
    st.code(", ".join(config.FEATURES), language="text")
    example = pd.DataFrame([{**{config.ID_COL: 1}, **{f: round(summary["feature_stats"][f]["50%"]) for f in config.FEATURES}}])
    st.download_button("Download CSV template", example.to_csv(index=False).encode("utf-8"),
                       "batch_template.csv", "text/csv")
    sum_low, sum_high = schema["feature_sum_range"]
    strict = st.checkbox("Reject rows outside the training range (any single feature, or the feature sum)",
                         value=False,
                         help=f"Off by default: such rows are still predicted but flagged as extrapolations. "
                              f"Training feature sums ranged from {sum_low} to {sum_high}.")
    upload = st.file_uploader("Upload CSV", type=["csv"])
    if upload is None:
        return
    try:
        shape, result, report, csv_bytes = run_batch(upload.getvalue(), strict)
    except Exception as exc:  # unreadable / malformed CSV
        st.error(f"Could not read the file as CSV: {exc}")
        return

    st.write(f"Uploaded **{shape[0]:,} rows × {shape[1]} columns**.")
    for err in report.blocking_errors:
        st.error(err)
    if not report.ok:
        return
    for warn in report.warnings:
        st.warning(warn)
    c = st.columns(4)
    c[0].metric("Rows predicted", f"{report.n_valid:,}")
    c[1].metric("Rows rejected", f"{len(report.rejected):,}")
    c[2].metric("Feature sum outside training range", f"{report.n_outside_joint_range:,}",
                help=f"Rows whose 20 features add up to less than {sum_low} or more than {sum_high}.")
    c[3].metric("Single feature outside training range", f"{report.n_outside_training_range:,}")
    if len(report.rejected):
        st.error("Some rows were rejected and have no prediction.")
        st.dataframe(report.rejected.head(200))
        st.download_button("Download rejected rows (CSV)", report.rejected.to_csv(index=False).encode("utf-8"),
                           "rejected_rows.csv", "text/csv")
    if len(result):
        st.subheader("Predictions")
        st.dataframe(format_predictions_for_display(result, max_rows=100))
        st.caption(f"Showing the first 100 rows with FloodProbability as a percentage (one decimal place). "
                   f"{DATASET_SCORE_NOTE} The download contains all predicted rows with the raw decimal "
                   "FloodProbability values (0-1 scale), unchanged, for compatibility with the submission format "
                   f"and other tools. The `{FLAG_COLUMN}` column is True when a row's feature sum or any single "
                   "feature is outside the training range (its prediction is an extrapolation).")
        st.download_button("Download predictions (CSV)", csv_bytes, "flood_predictions.csv", "text/csv",
                           type="primary")


def page_about():
    st.title("About the Project")
    st.subheader("Architecture")
    st.code(
        "data/*.csv --> src/data_processing.py --> src/train.py --> models/best_model.joblib\n"
        "                                                      +--> models/feature_schema.json\n"
        "                                                      +--> reports/metrics.json\n"
        "reports/ + models/ --> src/evaluate.py (figures, data summary)\n"
        "models/ --> src/predict.py --> app.py (Streamlit)   and   submissions/submission.csv",
        language="text")
    st.markdown(
        "Training code (`src/train.py`) is separate from the dashboard (`app.py`). The dashboard loads the saved "
        "model once and reads precomputed reports, so it never retrains and never loads the full dataset.")
    st.subheader("ML workflow")
    st.markdown("Load and validate → separate features from target (id excluded) → 80/20 split with a fixed seed → "
                "mean baseline → Linear Regression, Random Forest, Extra Trees, HistGradientBoosting → select on "
                "validation RMSE → save model + feature schema → evaluate with MAE, RMSE, R².")
    st.subheader("Dataset notes and limitations")
    st.markdown("- `train.csv` (1,117,957 rows) was used for training and validation.\n"
                "- `test.csv` has no target and was used only to create the submission file.\n"
                f"- Feature sums in training data ranged from {schema['feature_sum_range'][0]} to "
                f"{schema['feature_sum_range'][1]}. Inputs outside that joint range are flagged as extrapolations.\n"
                "- `flood.csv` follows an exact formula and has a different feature spread, so it was kept out of "
                "training and used only as a separate check.\n"
                "- No live sensors, timestamps or verified flood events: this is **not** real-time flood detection.")
    st.subheader("Technologies")
    st.markdown("Python, Pandas, NumPy, Scikit-learn, Matplotlib, Joblib, Streamlit.")
    st.subheader("Future improvements")
    st.markdown("- Integrate real rainfall, river-level and elevation data with timestamps.\n"
                "- Use verified historical flood events as labels and calibrate probabilities.\n"
                "- Train tree ensembles on all rows and tune them; try LightGBM or CatBoost.\n"
                "- Add geographic maps and user accounts, then evaluate time-aware validation.")


{"Home": page_home, "Flood Probability Prediction": page_predict, "Data Explorer": page_explorer,
 "Model Performance": page_performance, "Batch Prediction": page_batch, "About the Project": page_about}[page]()
