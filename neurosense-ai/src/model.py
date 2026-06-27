"""
model.py
========

Model training, evaluation and persistence for the NeuroSense AI demo.

Two classifiers are trained inside full scikit-learn pipelines (preprocessing +
estimator):

* Logistic Regression (with scaled numeric features)
* Random Forest Classifier (no scaling required)

The module computes accuracy / precision / recall / F1, the confusion matrix and
a classification report, then selects and persists the best-performing model to
``models/trained_model.pkl``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from .data_loader import RISK_LABELS
from .preprocessing import build_preprocessor

# Default location for the persisted best model.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
DEFAULT_MODEL_PATH = os.path.join(_PROJECT_ROOT, "models", "trained_model.pkl")


@dataclass
class ModelResult:
    """Container holding a fitted pipeline and its evaluation metrics."""

    name: str
    pipeline: Pipeline
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion: np.ndarray
    report_text: str
    report_dict: Dict = field(default_factory=dict)


def _make_logistic_regression() -> Pipeline:
    """Build a Logistic Regression pipeline (scaled numeric features)."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numeric=True)),
            (
                "classifier",
                LogisticRegression(max_iter=2000, random_state=42),
            ),
        ]
    )


def _make_random_forest() -> Pipeline:
    """Build a Random Forest pipeline (no scaling required)."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(scale_numeric=False)),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def _evaluate(name: str, pipeline: Pipeline, X_test, y_test) -> ModelResult:
    """Compute standard classification metrics for a fitted pipeline."""
    y_pred = pipeline.predict(X_test)
    target_names = [RISK_LABELS[i] for i in sorted(RISK_LABELS)]

    return ModelResult(
        name=name,
        pipeline=pipeline,
        accuracy=accuracy_score(y_test, y_pred),
        precision=precision_score(y_test, y_pred, average="weighted", zero_division=0),
        recall=recall_score(y_test, y_pred, average="weighted", zero_division=0),
        f1=f1_score(y_test, y_pred, average="weighted", zero_division=0),
        confusion=confusion_matrix(y_test, y_pred),
        report_text=classification_report(
            y_test, y_pred, target_names=target_names, zero_division=0
        ),
        report_dict=classification_report(
            y_test,
            y_pred,
            target_names=target_names,
            zero_division=0,
            output_dict=True,
        ),
    )


def train_models(X_train, X_test, y_train, y_test) -> Dict[str, ModelResult]:
    """Train and evaluate the Logistic Regression and Random Forest models.

    Returns:
        A dict mapping model name -> :class:`ModelResult`.
    """
    results: Dict[str, ModelResult] = {}

    lr = _make_logistic_regression()
    lr.fit(X_train, y_train)
    results["Logistic Regression"] = _evaluate("Logistic Regression", lr, X_test, y_test)

    rf = _make_random_forest()
    rf.fit(X_train, y_train)
    results["Random Forest"] = _evaluate("Random Forest", rf, X_test, y_test)

    return results


def select_best_model(results: Dict[str, ModelResult]) -> ModelResult:
    """Select the model with the highest weighted F1-score."""
    return max(results.values(), key=lambda r: r.f1)


def save_model(result: ModelResult, path: str = DEFAULT_MODEL_PATH) -> str:
    """Persist a fitted model (and its metadata) to disk with joblib.

    The pickled payload stores the pipeline plus a small metadata dict so the
    Prediction page can display which model is loaded and its headline metrics.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model_name": result.name,
        "pipeline": result.pipeline,
        "metrics": {
            "accuracy": result.accuracy,
            "precision": result.precision,
            "recall": result.recall,
            "f1": result.f1,
        },
    }
    joblib.dump(payload, path)
    return path


def load_model(path: str = DEFAULT_MODEL_PATH) -> Dict:
    """Load a persisted model payload from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No trained model found at: {path}")
    return joblib.load(path)


def predict_risk(payload: Dict, input_df: pd.DataFrame) -> Dict:
    """Predict the cognitive-risk class for a single input row.

    Args:
        payload: A loaded model payload (see :func:`load_model`).
        input_df: A one-row DataFrame containing all feature columns.

    Returns:
        Dict with predicted class index, label, confidence and per-class
        probabilities.
    """
    pipeline: Pipeline = payload["pipeline"]
    pred_class = int(pipeline.predict(input_df)[0])

    probabilities = {}
    confidence = None
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(input_df)[0]
        classes = pipeline.classes_
        probabilities = {int(c): float(p) for c, p in zip(classes, proba)}
        confidence = float(np.max(proba))

    return {
        "class_index": pred_class,
        "label": RISK_LABELS.get(pred_class, str(pred_class)),
        "confidence": confidence,
        "probabilities": probabilities,
    }
