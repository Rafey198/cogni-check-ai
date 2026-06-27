"""
preprocessing.py
================

Data-preparation utilities for the NeuroSense AI demo.

Responsibilities:
* Handle missing values (median for numeric, most-frequent for categorical).
* Encode categorical columns (one-hot for the ``gender`` field).
* Scale numeric features (used by Logistic Regression).
* Split the data into train/test partitions.

The functions return scikit-learn compatible objects so the same preprocessing
can be reused at prediction time.
"""

from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data_loader import FEATURE_COLUMNS, TARGET_COLUMN

# Columns that are categorical and therefore require encoding.
CATEGORICAL_COLUMNS = ["gender"]

# All remaining feature columns are numeric.
NUMERIC_COLUMNS = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_COLUMNS]


def build_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    """Create a ``ColumnTransformer`` that imputes, scales and encodes features.

    Args:
        scale_numeric: When True numeric features are standardized. Tree-based
            models (Random Forest) do not require scaling, so this can be set
            to False for them.

    Returns:
        A scikit-learn ``ColumnTransformer`` ready to be fit.
    """
    # Numeric pipeline: median imputation (+ optional standard scaling).
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(steps=numeric_steps)

    # Categorical pipeline: most-frequent imputation + one-hot encoding.
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            # ``handle_unknown='ignore'`` keeps prediction robust to unseen labels.
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_COLUMNS),
            ("cat", categorical_pipeline, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
    )
    return preprocessor


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split a dataset into train/test feature and target partitions.

    Args:
        df: The full dataset including the target column.
        test_size: Fraction of rows reserved for testing.
        random_state: Seed for reproducibility.

    Returns:
        ``(X_train, X_test, y_train, y_test)``.
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")

    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].astype(int)

    # Stratify on the label so all risk classes appear in both partitions.
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def basic_quality_report(df: pd.DataFrame) -> Dict[str, object]:
    """Return a small dictionary summarising dataset quality.

    Used by the Dataset/Preprocessing UI to show missing-value counts etc.
    """
    return {
        "n_rows": int(len(df)),
        "n_features": int(len([c for c in FEATURE_COLUMNS if c in df.columns])),
        "missing_per_column": df.isna().sum().to_dict(),
        "total_missing": int(df.isna().sum().sum()),
    }
