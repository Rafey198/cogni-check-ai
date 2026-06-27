"""
explainability.py
==================

Lightweight model-explainability helpers for the NeuroSense AI demo.

The main capability is extracting Random Forest feature importances and mapping
them back to the original (human-readable) feature names. Because the model is
wrapped in a ColumnTransformer pipeline, we resolve the transformed feature
names and then aggregate one-hot encoded columns back to their source feature.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from .preprocessing import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS


def _get_transformed_feature_names(pipeline: Pipeline) -> list:
    """Return the feature names produced by the pipeline's preprocessor."""
    preprocessor = pipeline.named_steps["preprocessor"]
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        # Fallback for older sklearn versions: build names manually.
        names = list(NUMERIC_COLUMNS)
        names += [f"cat__{c}" for c in CATEGORICAL_COLUMNS]
        return names


def get_feature_importances(pipeline: Pipeline) -> Optional[pd.DataFrame]:
    """Extract aggregated feature importances from a Random Forest pipeline.

    Args:
        pipeline: A fitted sklearn pipeline whose classifier exposes
            ``feature_importances_`` (e.g. a Random Forest).

    Returns:
        A DataFrame with columns ``feature`` and ``importance`` sorted in
        descending order, or ``None`` if the model does not support importances.
    """
    classifier = pipeline.named_steps.get("classifier")
    if not isinstance(classifier, RandomForestClassifier) and not hasattr(
        classifier, "feature_importances_"
    ):
        return None

    importances = classifier.feature_importances_
    transformed_names = _get_transformed_feature_names(pipeline)

    if len(transformed_names) != len(importances):
        # Safety guard: lengths must align.
        transformed_names = [f"feature_{i}" for i in range(len(importances))]

    raw = pd.DataFrame({"raw_feature": transformed_names, "importance": importances})

    # Map transformed names back to the original source feature so one-hot
    # encoded gender columns are aggregated into a single "gender" importance.
    def _source_feature(name: str) -> str:
        clean = name.split("__")[-1] if "__" in name else name
        for base in NUMERIC_COLUMNS + CATEGORICAL_COLUMNS:
            if clean == base or clean.startswith(base + "_"):
                return base
        return clean

    raw["feature"] = raw["raw_feature"].apply(_source_feature)
    agg = (
        raw.groupby("feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return agg


def top_features_summary(importance_df: pd.DataFrame, top_n: int = 3) -> str:
    """Create a short natural-language summary of the most important features."""
    if importance_df is None or importance_df.empty:
        return "Feature importance is not available for the selected model."

    top = importance_df.head(top_n)["feature"].tolist()
    pretty = ", ".join(f"`{f}`" for f in top)
    return (
        f"In this research-demo model, the features contributing most to the "
        f"risk prediction are {pretty}. These drivers reflect patterns in the "
        f"synthetic dataset and should not be interpreted as clinical evidence."
    )
