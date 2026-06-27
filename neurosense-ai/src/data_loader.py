"""
data_loader.py
==============

Utilities for loading the multimodal cognitive-risk dataset used by the
NeuroSense AI research demo.

The module is intentionally self-contained so the Streamlit app works out of
the box on Streamlit Community Cloud without requiring any external download:

* If ``data/sample_cognitive_data.csv`` exists, it is loaded directly.
* If it is missing, a realistic synthetic dataset is generated on the fly and
  (when possible) cached to disk so subsequent runs are fast.

NOTE: The dataset is fully synthetic. It is designed only to demonstrate an
end-to-end machine-learning workflow and must never be used for any clinical
or diagnostic purpose.
"""

from __future__ import annotations

import os
from typing import List

import numpy as np
import pandas as pd

# Ordered list of feature columns expected by the rest of the application.
FEATURE_COLUMNS: List[str] = [
    "age",
    "gender",
    "education_years",
    "memory_score",
    "attention_score",
    "reaction_time",
    "sleep_hours",
    "activity_level",
    "walking_stability",
    "speech_pause_rate",
    "lexical_diversity",
    "mmse_score",
]

# Name of the supervised target column.
TARGET_COLUMN: str = "cognitive_risk_label"

# Human-readable names for the three risk classes.
RISK_LABELS = {0: "Low Risk", 1: "Moderate Risk", 2: "High Risk"}

# Default location of the bundled sample dataset (relative to project root).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
DEFAULT_DATA_PATH = os.path.join(_PROJECT_ROOT, "data", "sample_cognitive_data.csv")


def generate_sample_dataset(n_samples: int = 600, random_state: int = 42) -> pd.DataFrame:
    """Generate a realistic *synthetic* multimodal cognitive-risk dataset.

    The data-generating process loosely mimics relationships seen in cognitive
    screening research (e.g. lower memory/MMSE scores, slower reaction times and
    higher speech-pause rates trend toward higher risk). It is purely
    illustrative and contains no real patient data.

    Args:
        n_samples: Number of synthetic participants to create.
        random_state: Seed for reproducibility.

    Returns:
        A pandas DataFrame containing the feature columns plus the target label.
    """
    rng = np.random.default_rng(random_state)

    # --- Demographic / clinical-style features -------------------------------
    age = rng.normal(65, 12, n_samples).clip(40, 95)
    gender = rng.choice(["Male", "Female"], size=n_samples)
    education_years = rng.normal(13, 3.5, n_samples).clip(2, 22)

    # --- Cognitive test features --------------------------------------------
    # Memory and attention scores (0-100, higher is better).
    memory_score = rng.normal(75, 15, n_samples).clip(0, 100)
    attention_score = rng.normal(72, 16, n_samples).clip(0, 100)

    # --- Behavioral / sensor features ---------------------------------------
    reaction_time = rng.normal(450, 120, n_samples).clip(180, 1200)  # milliseconds
    sleep_hours = rng.normal(6.8, 1.3, n_samples).clip(3, 11)
    activity_level = rng.normal(55, 20, n_samples).clip(0, 100)  # 0-100 index
    walking_stability = rng.normal(78, 14, n_samples).clip(0, 100)  # gait stability index

    # --- Speech / language features -----------------------------------------
    speech_pause_rate = rng.normal(0.18, 0.07, n_samples).clip(0.01, 0.6)  # pauses per sec
    lexical_diversity = rng.normal(0.65, 0.12, n_samples).clip(0.2, 0.95)  # type-token ratio

    # --- MMSE (Mini-Mental State Examination style score, 0-30) -------------
    mmse_score = rng.normal(27, 3.5, n_samples).clip(5, 30)

    # ------------------------------------------------------------------------
    # Build a latent "risk" signal as a weighted combination of features.
    # Higher latent score -> higher cognitive risk.
    # ------------------------------------------------------------------------
    z = (
        0.030 * (age - 65)
        - 0.025 * (memory_score - 75)
        - 0.022 * (attention_score - 72)
        + 0.006 * (reaction_time - 450)
        - 0.020 * (mmse_score - 27) * 3
        - 0.015 * (education_years - 13)
        + 4.0 * (speech_pause_rate - 0.18)
        - 2.5 * (lexical_diversity - 0.65)
        - 0.015 * (walking_stability - 78)
        - 0.010 * (activity_level - 55)
        - 0.05 * (sleep_hours - 6.8)
    )

    # Add noise so the problem is non-trivial.
    z = z + rng.normal(0, 0.8, n_samples)

    # Convert the continuous latent score into 3 ordinal risk classes using
    # empirical tertiles so the classes are reasonably balanced.
    low_cut, high_cut = np.quantile(z, [0.40, 0.75])
    labels = np.where(z <= low_cut, 0, np.where(z <= high_cut, 1, 2))

    df = pd.DataFrame(
        {
            "age": np.round(age, 1),
            "gender": gender,
            "education_years": np.round(education_years, 1),
            "memory_score": np.round(memory_score, 1),
            "attention_score": np.round(attention_score, 1),
            "reaction_time": np.round(reaction_time, 1),
            "sleep_hours": np.round(sleep_hours, 1),
            "activity_level": np.round(activity_level, 1),
            "walking_stability": np.round(walking_stability, 1),
            "speech_pause_rate": np.round(speech_pause_rate, 3),
            "lexical_diversity": np.round(lexical_diversity, 3),
            "mmse_score": np.round(mmse_score, 1),
            TARGET_COLUMN: labels.astype(int),
        }
    )

    # Inject a small number of missing values to exercise the preprocessing
    # pipeline (so the demo realistically handles imputation).
    for col in ["sleep_hours", "activity_level", "lexical_diversity"]:
        missing_idx = rng.choice(n_samples, size=max(1, n_samples // 60), replace=False)
        df.loc[missing_idx, col] = np.nan

    return df


def load_data(path: str = DEFAULT_DATA_PATH, generate_if_missing: bool = True) -> pd.DataFrame:
    """Load the cognitive-risk dataset from ``path``.

    If the file does not exist and ``generate_if_missing`` is True, a synthetic
    dataset is generated and (best-effort) written to ``path``.

    Args:
        path: Path to a CSV dataset.
        generate_if_missing: Whether to generate a synthetic dataset when the
            file is absent.

    Returns:
        The loaded (or generated) dataset as a DataFrame.
    """
    if os.path.exists(path):
        return pd.read_csv(path)

    if not generate_if_missing:
        raise FileNotFoundError(f"Dataset not found at: {path}")

    df = generate_sample_dataset()
    # Best-effort caching to disk (Streamlit Cloud allows ephemeral writes).
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
    except OSError:
        pass
    return df


def get_feature_metadata() -> pd.DataFrame:
    """Return a small descriptor table documenting each feature.

    Used by the Dataset page to explain the columns to the user.
    """
    rows = [
        ("age", "Demographic", "Participant age in years"),
        ("gender", "Demographic", "Self-reported gender (Male/Female)"),
        ("education_years", "Demographic", "Total years of formal education"),
        ("memory_score", "Cognitive", "Memory recall test score (0-100)"),
        ("attention_score", "Cognitive", "Sustained attention test score (0-100)"),
        ("reaction_time", "Behavioral", "Mean reaction time in milliseconds"),
        ("sleep_hours", "Behavioral", "Average nightly sleep duration (hours)"),
        ("activity_level", "Behavioral", "Daily physical activity index (0-100)"),
        ("walking_stability", "Sensor/Gait", "Gait/balance stability index (0-100)"),
        ("speech_pause_rate", "Speech", "Pauses per second during speech tasks"),
        ("lexical_diversity", "Speech", "Type-token ratio of spoken language (0-1)"),
        ("mmse_score", "Clinical", "Mini-Mental State Examination style score (0-30)"),
        (TARGET_COLUMN, "Target", "0=Low Risk, 1=Moderate Risk, 2=High Risk"),
    ]
    return pd.DataFrame(rows, columns=["feature", "modality", "description"])
