"""
NeuroSense AI - Streamlit Research Demo
=======================================

A Multimodal Digital Biomarker System for Early Cognitive Risk Screening.

IMPORTANT: This is a research/academic demonstration only. It is NOT a medical
diagnosis or clinical decision system. The bundled dataset is fully synthetic.

Run locally with:
    streamlit run app.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

# Ensure the project root is importable so ``src`` works both locally and on
# Streamlit Community Cloud regardless of the current working directory.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import (  # noqa: E402
    DEFAULT_DATA_PATH,
    FEATURE_COLUMNS,
    RISK_LABELS,
    TARGET_COLUMN,
    get_feature_metadata,
    load_data,
)
from src.explainability import (  # noqa: E402
    get_feature_importances,
    top_features_summary,
)
from src.model import (  # noqa: E402
    DEFAULT_MODEL_PATH,
    load_model,
    predict_risk,
    save_model,
    select_best_model,
    train_models,
)
from src.preprocessing import basic_quality_report, split_data  # noqa: E402
from src.report import (  # noqa: E402
    DISCLAIMER,
    LIMITATIONS,
    PROJECT_TITLE,
    generate_report,
)

# ---------------------------------------------------------------------------
# Page configuration & global styling
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NeuroSense AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# A little CSS for a clean, professional look.
st.markdown(
    """
    <style>
        .main > div { padding-top: 1.5rem; }
        .ns-hero {
            background: linear-gradient(135deg, #1e3a8a 0%, #2563eb 100%);
            padding: 1.6rem 2rem; border-radius: 14px; color: #ffffff;
            margin-bottom: 1.2rem;
        }
        .ns-hero h1 { color: #ffffff; margin-bottom: .3rem; font-size: 1.7rem; }
        .ns-hero p { color: #e0e7ff; margin: 0; }
        .ns-card {
            background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px;
            padding: 1rem 1.2rem; margin-bottom: 1rem;
        }
        .ns-disclaimer {
            background: #fff7ed; border-left: 5px solid #f97316;
            padding: .8rem 1rem; border-radius: 8px; color: #7c2d12;
            font-size: .92rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

DISCLAIMER_HTML = f'<div class="ns-disclaimer"><strong>⚠️ Disclaimer:</strong> {DISCLAIMER}</div>'


# ---------------------------------------------------------------------------
# Cached resource helpers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _cached_dataset(path: str) -> pd.DataFrame:
    """Load (or generate) the bundled dataset, cached for performance."""
    return load_data(path)


def _get_dataset() -> pd.DataFrame:
    """Return the active dataset (uploaded one if present, else the sample)."""
    if "uploaded_df" in st.session_state and st.session_state["uploaded_df"] is not None:
        return st.session_state["uploaded_df"]
    return _cached_dataset(DEFAULT_DATA_PATH)


def _class_distribution(df: pd.DataFrame) -> dict:
    """Return a readable label -> count mapping for the target column."""
    if TARGET_COLUMN not in df.columns:
        return {}
    counts = df[TARGET_COLUMN].value_counts().sort_index()
    return {RISK_LABELS.get(int(k), str(k)): int(v) for k, v in counts.items()}


def _try_load_model():
    """Load the persisted model payload if available, else None."""
    try:
        return load_model(DEFAULT_MODEL_PATH)
    except FileNotFoundError:
        return None


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("🧠 NeuroSense AI")
st.sidebar.caption("Multimodal Digital Biomarker Research Demo")
PAGES = [
    "Home",
    "Dataset",
    "Train Model",
    "Risk Prediction",
    "Explainability",
    "Research Report",
    "About",
]
page = st.sidebar.radio("Navigation", PAGES, label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.info("Research prototype • Not for clinical use")


# ===========================================================================
# PAGE: HOME
# ===========================================================================
def render_home() -> None:
    st.markdown(
        f"""
        <div class="ns-hero">
            <h1>🧠 {PROJECT_TITLE}</h1>
            <p>An end-to-end research prototype for early cognitive risk screening
            using multimodal behavioral, cognitive and clinical-style features.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(DISCLAIMER_HTML, unsafe_allow_html=True)
    st.write("")

    col1, col2 = st.columns([1.4, 1])
    with col1:
        st.subheader("What does this demo do?")
        st.markdown(
            """
            NeuroSense AI illustrates a complete machine-learning workflow for
            **cognitive risk screening research**:

            - **Multimodal features** — combines cognitive tests, behavioral
              signals, speech/language metrics, gait stability and a clinical-style
              MMSE score.
            - **Transparent modeling** — trains Logistic Regression and Random
              Forest classifiers and compares their performance.
            - **Explainability** — surfaces which features drive predictions.
            - **Reporting** — exports a concise, downloadable research report.

            The target distinguishes three classes: **Low**, **Moderate** and
            **High** cognitive risk.
            """
        )
    with col2:
        st.subheader("Quick start")
        st.markdown(
            """
            1. **Dataset** — explore the synthetic data or upload your own CSV.
            2. **Train Model** — fit and compare models.
            3. **Risk Prediction** — try the interactive prediction form.
            4. **Explainability** — inspect feature importance.
            5. **Research Report** — download a summary.
            """
        )

    st.markdown("### Pipeline at a glance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Risk classes", "3")
    c2.metric("Feature modalities", "5")
    c3.metric("Models compared", "2")
    c4.metric("Input features", str(len(FEATURE_COLUMNS)))


# ===========================================================================
# PAGE: DATASET
# ===========================================================================
def render_dataset() -> None:
    st.header("📊 Dataset")
    st.write(
        "Explore the bundled **synthetic** multimodal cognitive-risk dataset, "
        "or upload your own CSV with the same columns."
    )

    # --- Upload widget ------------------------------------------------------
    uploaded = st.file_uploader("Upload a custom CSV dataset (optional)", type=["csv"])
    if uploaded is not None:
        try:
            df_up = pd.read_csv(uploaded)
            st.session_state["uploaded_df"] = df_up
            st.success(f"Loaded uploaded dataset with {len(df_up)} rows.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not read the uploaded file: {exc}")

    if st.session_state.get("uploaded_df") is not None:
        if st.button("↩️ Reset to bundled sample dataset"):
            st.session_state["uploaded_df"] = None
            st.rerun()

    df = _get_dataset()

    # --- Overview metrics ---------------------------------------------------
    quality = basic_quality_report(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", quality["n_rows"])
    c2.metric("Features", quality["n_features"])
    c3.metric("Missing values", quality["total_missing"])
    c4.metric("Target classes", df[TARGET_COLUMN].nunique() if TARGET_COLUMN in df else "N/A")

    st.markdown("#### Data preview")
    st.dataframe(df.head(20), use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Class distribution")
        dist = _class_distribution(df)
        if dist:
            st.bar_chart(pd.Series(dist))
        else:
            st.info("Target column not found in this dataset.")
    with col_b:
        st.markdown("#### Summary statistics")
        numeric_df = df.select_dtypes(include=[np.number])
        st.dataframe(numeric_df.describe().T, use_container_width=True)

    st.markdown("#### Feature dictionary")
    st.dataframe(get_feature_metadata(), use_container_width=True)

    with st.expander("Missing values per column"):
        st.write(quality["missing_per_column"])


# ===========================================================================
# PAGE: TRAIN MODEL
# ===========================================================================
def render_train_model() -> None:
    st.header("🤖 Train Model")
    st.write(
        "Train and compare two classifiers — **Logistic Regression** and "
        "**Random Forest** — then save the best model for prediction."
    )

    df = _get_dataset()
    if TARGET_COLUMN not in df.columns:
        st.error(
            f"The active dataset has no '{TARGET_COLUMN}' column, so it cannot be "
            "used for supervised training. Upload a labelled dataset or reset to "
            "the bundled sample."
        )
        return

    test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)

    if st.button("🚀 Train models", type="primary"):
        with st.spinner("Training models..."):
            X_train, X_test, y_train, y_test = split_data(df, test_size=test_size)
            results = train_models(X_train, X_test, y_train, y_test)
            best = select_best_model(results)
            save_model(best, DEFAULT_MODEL_PATH)

            # Persist results in session for other pages.
            st.session_state["train_results"] = results
            st.session_state["best_model_name"] = best.name
            st.session_state["best_metrics"] = {
                "accuracy": best.accuracy,
                "precision": best.precision,
                "recall": best.recall,
                "f1": best.f1,
            }
        st.success(f"Training complete. Best model: **{best.name}** (saved to disk).")

    results = st.session_state.get("train_results")
    if not results:
        st.info("Click **Train models** to fit and evaluate the classifiers.")
        return

    # --- Comparison table ---------------------------------------------------
    st.markdown("#### Model comparison")
    comparison = pd.DataFrame(
        [
            {
                "Model": r.name,
                "Accuracy": r.accuracy,
                "Precision": r.precision,
                "Recall": r.recall,
                "F1-score": r.f1,
            }
            for r in results.values()
        ]
    ).set_index("Model")
    st.dataframe(comparison.style.format("{:.4f}").highlight_max(axis=0, color="#bbf7d0"),
                 use_container_width=True)

    best_name = st.session_state.get("best_model_name")
    st.success(f"Selected best model (highest F1): **{best_name}**")

    # --- Per-model detail ---------------------------------------------------
    for name, r in results.items():
        with st.expander(f"Details — {name}", expanded=(name == best_name)):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Accuracy", f"{r.accuracy:.3f}")
            m2.metric("Precision", f"{r.precision:.3f}")
            m3.metric("Recall", f"{r.recall:.3f}")
            m4.metric("F1-score", f"{r.f1:.3f}")

            st.markdown("**Confusion matrix**")
            labels = [RISK_LABELS[i] for i in sorted(RISK_LABELS)]
            cm_df = pd.DataFrame(
                r.confusion,
                index=[f"Actual: {l}" for l in labels],
                columns=[f"Pred: {l}" for l in labels],
            )
            st.dataframe(cm_df, use_container_width=True)

            st.markdown("**Classification report**")
            st.code(r.report_text)


# ===========================================================================
# PAGE: RISK PREDICTION
# ===========================================================================
def render_prediction() -> None:
    st.header("🔮 Risk Prediction")
    st.markdown(DISCLAIMER_HTML, unsafe_allow_html=True)
    st.write("")

    payload = _try_load_model()
    if payload is None:
        st.warning(
            "No trained model found. Please visit the **Train Model** page and "
            "train a model first."
        )
        return

    st.caption(
        f"Using saved model: **{payload.get('model_name', 'Unknown')}**"
    )

    df = _get_dataset()
    # Use dataset medians as sensible default form values.
    numeric_defaults = df.select_dtypes(include=[np.number]).median(numeric_only=True)

    st.markdown("#### Enter participant features")
    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age (years)", 40.0, 95.0,
                                  float(numeric_defaults.get("age", 65)), 1.0)
            gender = st.selectbox("Gender", ["Male", "Female"])
            education_years = st.number_input("Education (years)", 2.0, 22.0,
                                              float(numeric_defaults.get("education_years", 13)), 0.5)
            memory_score = st.slider("Memory score (0-100)", 0.0, 100.0,
                                     float(numeric_defaults.get("memory_score", 75)))
        with c2:
            attention_score = st.slider("Attention score (0-100)", 0.0, 100.0,
                                        float(numeric_defaults.get("attention_score", 72)))
            reaction_time = st.number_input("Reaction time (ms)", 180.0, 1200.0,
                                            float(numeric_defaults.get("reaction_time", 450)), 10.0)
            sleep_hours = st.slider("Sleep (hours)", 3.0, 11.0,
                                    float(numeric_defaults.get("sleep_hours", 6.8)), 0.1)
            activity_level = st.slider("Activity level (0-100)", 0.0, 100.0,
                                       float(numeric_defaults.get("activity_level", 55)))
        with c3:
            walking_stability = st.slider("Walking stability (0-100)", 0.0, 100.0,
                                          float(numeric_defaults.get("walking_stability", 78)))
            speech_pause_rate = st.slider("Speech pause rate (pauses/sec)", 0.01, 0.6,
                                          float(numeric_defaults.get("speech_pause_rate", 0.18)), 0.01)
            lexical_diversity = st.slider("Lexical diversity (0-1)", 0.2, 0.95,
                                          float(numeric_defaults.get("lexical_diversity", 0.65)), 0.01)
            mmse_score = st.slider("MMSE score (0-30)", 5.0, 30.0,
                                   float(numeric_defaults.get("mmse_score", 27)), 0.5)

        submitted = st.form_submit_button("Predict risk", type="primary")

    if not submitted:
        st.info("Adjust the inputs and click **Predict risk**.")
        return

    # Assemble a one-row DataFrame in the exact feature order.
    input_row = {
        "age": age,
        "gender": gender,
        "education_years": education_years,
        "memory_score": memory_score,
        "attention_score": attention_score,
        "reaction_time": reaction_time,
        "sleep_hours": sleep_hours,
        "activity_level": activity_level,
        "walking_stability": walking_stability,
        "speech_pause_rate": speech_pause_rate,
        "lexical_diversity": lexical_diversity,
        "mmse_score": mmse_score,
    }
    input_df = pd.DataFrame([input_row])[FEATURE_COLUMNS]

    result = predict_risk(payload, input_df)
    label = result["label"]
    confidence = result["confidence"]

    interpretation = _build_interpretation(label)

    # Persist for the Research Report page.
    st.session_state["last_prediction"] = {
        "label": label,
        "confidence": confidence,
        "interpretation": interpretation,
    }

    st.markdown("### Result")
    color = {"Low Risk": "🟢", "Moderate Risk": "🟡", "High Risk": "🔴"}.get(label, "⚪")
    rc1, rc2 = st.columns([1, 1])
    with rc1:
        st.metric("Predicted category", f"{color} {label}")
    with rc2:
        if confidence is not None:
            st.metric("Confidence score", f"{confidence * 100:.1f}%")

    if result.get("probabilities"):
        prob_series = pd.Series(
            {RISK_LABELS.get(k, str(k)): v for k, v in result["probabilities"].items()}
        )
        st.markdown("**Class probabilities**")
        st.bar_chart(prob_series)

    st.markdown("#### Interpretation")
    st.info(interpretation)
    st.markdown(DISCLAIMER_HTML, unsafe_allow_html=True)


def _build_interpretation(label: str) -> str:
    """Return a careful, non-medical interpretation string for a risk label."""
    base = {
        "Low Risk": (
            "The entered pattern is closer to the low-risk group in this research "
            "demo dataset."
        ),
        "Moderate Risk": (
            "The entered pattern is closer to the moderate-risk group in this "
            "research demo dataset."
        ),
        "High Risk": (
            "The entered pattern is closer to the high-risk group in this research "
            "demo dataset."
        ),
    }.get(label, "The entered pattern was classified by the research demo model.")
    return (
        base
        + " Clinical confirmation would require assessment by a qualified "
        "healthcare professional. This output is not a diagnosis."
    )


# ===========================================================================
# PAGE: EXPLAINABILITY
# ===========================================================================
def render_explainability() -> None:
    st.header("💡 Explainability")
    st.write(
        "Understand which features the model relies on most. Feature importance "
        "is shown for the **Random Forest** model."
    )

    # Prefer a freshly trained RF in session; otherwise fall back to the saved
    # model if it is a Random Forest.
    rf_pipeline = None
    results = st.session_state.get("train_results")
    if results and "Random Forest" in results:
        rf_pipeline = results["Random Forest"].pipeline
    else:
        payload = _try_load_model()
        if payload and payload.get("model_name") == "Random Forest":
            rf_pipeline = payload["pipeline"]

    if rf_pipeline is None:
        st.warning(
            "No Random Forest model is available yet. Please train models on the "
            "**Train Model** page (the Random Forest is always trained)."
        )
        return

    importance_df = get_feature_importances(rf_pipeline)
    if importance_df is None or importance_df.empty:
        st.error("Could not compute feature importances for this model.")
        return

    # --- Plot (matplotlib) --------------------------------------------------
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_df = importance_df.sort_values("importance", ascending=True)
    ax.barh(plot_df["feature"], plot_df["importance"], color="#2563eb")
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest Feature Importance")
    fig.tight_layout()
    st.pyplot(fig)

    st.markdown("#### Ranked feature importance")
    st.dataframe(
        importance_df.rename(columns={"feature": "Feature", "importance": "Importance"}),
        use_container_width=True,
    )

    st.markdown("#### What this means")
    st.info(top_features_summary(importance_df))


# ===========================================================================
# PAGE: RESEARCH REPORT
# ===========================================================================
def render_report() -> None:
    st.header("📄 Research Report")
    st.write(
        "Generate and download a concise text report summarising the current "
        "session: dataset, model, metrics, prediction, limitations and disclaimer."
    )

    df = _get_dataset()
    dataset_summary = {
        "n_rows": int(len(df)),
        "n_features": len([c for c in FEATURE_COLUMNS if c in df.columns]),
        "class_distribution": _class_distribution(df),
    }

    model_name = st.session_state.get("best_model_name")
    metrics = st.session_state.get("best_metrics")

    if model_name is None:
        # Fall back to the saved model payload, if any.
        payload = _try_load_model()
        if payload:
            model_name = payload.get("model_name")
            metrics = payload.get("metrics")

    if model_name is None:
        st.warning(
            "No trained model found. Train a model first to include performance "
            "metrics in the report."
        )

    prediction = st.session_state.get("last_prediction")

    report_text = generate_report(
        dataset_summary=dataset_summary,
        model_name=model_name or "Not trained",
        metrics=metrics or {},
        prediction=prediction,
    )

    st.markdown("#### Report preview")
    st.code(report_text)

    st.download_button(
        label="⬇️ Download report (.txt)",
        data=report_text,
        file_name="neurosense_ai_research_report.txt",
        mime="text/plain",
    )


# ===========================================================================
# PAGE: ABOUT
# ===========================================================================
def render_about() -> None:
    st.header("ℹ️ About")
    st.markdown(
        f"""
        **{PROJECT_TITLE}** is a research prototype inspired by AI-based cognitive
        screening concepts that combine **multimodal behavioral, cognitive and
        clinical-style features** — including memory and attention tests, reaction
        time, sleep and activity patterns, gait/walking stability, and speech/language
        metrics such as pause rate and lexical diversity.

        ### Purpose
        The goal is purely educational: to demonstrate a clean, end-to-end machine
        learning workflow (data → preprocessing → modeling → explainability →
        reporting) within a professional Streamlit interface.

        ### How it could be extended
        In a real research program, this prototype could be extended with:

        - **Real speech/audio processing** — extracting acoustic and linguistic
          biomarkers from raw recordings.
        - **Computer vision** — facial/gait analysis from video.
        - **Wearable sensor data** — continuous activity, sleep and heart-rate
          signals from devices.
        - **Clinical validation** — prospective studies, IRB approval, and
          comparison against gold-standard neuropsychological assessments.

        ### Technology
        Built with Python, Streamlit, pandas, NumPy, scikit-learn, matplotlib and
        joblib.
        """
    )
    st.markdown(DISCLAIMER_HTML, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
PAGE_RENDERERS = {
    "Home": render_home,
    "Dataset": render_dataset,
    "Train Model": render_train_model,
    "Risk Prediction": render_prediction,
    "Explainability": render_explainability,
    "Research Report": render_report,
    "About": render_about,
}

PAGE_RENDERERS[page]()
