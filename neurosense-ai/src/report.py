"""
report.py
=========

Generates a short, downloadable plain-text research report summarising a
NeuroSense AI demo session: dataset summary, selected model, performance
metrics, the latest prediction, limitations and the mandatory disclaimer.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

PROJECT_TITLE = (
    "NeuroSense AI: A Multimodal Digital Biomarker System "
    "for Early Cognitive Risk Screening"
)

DISCLAIMER = (
    "This tool is for academic/research demonstration only and is not a medical "
    "diagnosis or clinical decision system."
)

LIMITATIONS = [
    "The dataset is fully synthetic and does not represent real patients.",
    "No clinical validation, regulatory approval, or peer review has been performed.",
    "Speech, gait and sensor features are simulated, not extracted from raw signals.",
    "Predictions reflect patterns in demo data only and must not guide care.",
    "Model performance will differ substantially on real-world clinical data.",
]


def generate_report(
    dataset_summary: Dict,
    model_name: str,
    metrics: Dict,
    prediction: Optional[Dict] = None,
) -> str:
    """Build the full text report as a single string.

    Args:
        dataset_summary: Dict with keys like ``n_rows``, ``n_features`` and
            optionally ``class_distribution``.
        model_name: Name of the selected/best model.
        metrics: Dict of headline metrics (accuracy/precision/recall/f1).
        prediction: Optional dict describing the most recent prediction.

    Returns:
        The report contents as a formatted string.
    """
    lines = []
    sep = "=" * 70

    lines.append(sep)
    lines.append(PROJECT_TITLE)
    lines.append("Research Demonstration Report")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(sep)
    lines.append("")

    # --- Dataset summary ----------------------------------------------------
    lines.append("1. DATASET SUMMARY")
    lines.append("-" * 70)
    lines.append(f"Number of records : {dataset_summary.get('n_rows', 'N/A')}")
    lines.append(f"Number of features: {dataset_summary.get('n_features', 'N/A')}")
    class_dist = dataset_summary.get("class_distribution")
    if class_dist:
        lines.append("Class distribution:")
        for label, count in class_dist.items():
            lines.append(f"   - {label}: {count}")
    lines.append("")

    # --- Selected model -----------------------------------------------------
    lines.append("2. SELECTED MODEL")
    lines.append("-" * 70)
    lines.append(f"Best model: {model_name}")
    lines.append("")

    # --- Performance metrics ------------------------------------------------
    lines.append("3. PERFORMANCE METRICS (test set)")
    lines.append("-" * 70)
    if metrics:
        lines.append(f"Accuracy : {metrics.get('accuracy', 0):.4f}")
        lines.append(f"Precision: {metrics.get('precision', 0):.4f}")
        lines.append(f"Recall   : {metrics.get('recall', 0):.4f}")
        lines.append(f"F1-score : {metrics.get('f1', 0):.4f}")
    else:
        lines.append("Metrics not available (model not yet trained in this session).")
    lines.append("")

    # --- Prediction ---------------------------------------------------------
    lines.append("4. PREDICTED RISK CATEGORY")
    lines.append("-" * 70)
    if prediction:
        lines.append(f"Predicted category: {prediction.get('label', 'N/A')}")
        conf = prediction.get("confidence")
        if conf is not None:
            lines.append(f"Confidence score  : {conf * 100:.1f}%")
        interp = prediction.get("interpretation")
        if interp:
            lines.append(f"Interpretation    : {interp}")
    else:
        lines.append("No prediction has been made in this session yet.")
    lines.append("")

    # --- Limitations --------------------------------------------------------
    lines.append("5. LIMITATIONS")
    lines.append("-" * 70)
    for item in LIMITATIONS:
        lines.append(f" - {item}")
    lines.append("")

    # --- Disclaimer ---------------------------------------------------------
    lines.append("6. DISCLAIMER")
    lines.append("-" * 70)
    lines.append(DISCLAIMER)
    lines.append("")
    lines.append(sep)

    return "\n".join(lines)
