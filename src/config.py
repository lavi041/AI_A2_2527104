"""Project-wide constants and identity helpers (student id, seed, verification code)."""
from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
PLOTS_DIR = ARTIFACTS_DIR / "plots"

TEST_SIZE = 0.25
CV_FOLDS = 5
K_RANGE = range(2, 7)  # K = 2..6

# Feature ranges = ranges used by the data generator. Also used for input validation.
NUMERIC_RANGES = {
    "rainfall_mm": (600.0, 1400.0),
    "avg_temp_c": (12.0, 26.0),
    "soil_moisture_pct": (20.0, 80.0),
    "fertilizer_kg": (0.0, 250.0),
    "field_size_ha": (0.2, 4.0),
}
NUMERIC_FEATURES = list(NUMERIC_RANGES)
CATEGORICAL_FEATURES = ["season"]
BINARY_FEATURES = ["cooperative_support"]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BINARY_FEATURES

REGRESSION_TARGET = "yield_tons"
CLASSIFICATION_TARGET = "high_risk"
REQUIRED_COLUMNS = FEATURE_COLUMNS + [REGRESSION_TARGET, CLASSIFICATION_TARGET]
# Columns that must never be model inputs (target leakage).
FORBIDDEN_FEATURES = ["yield_tons", "yield_per_ha", "high_risk"]

DISCLAIMER = (
    "Classroom decision-support prototype trained on SYNTHETIC data. It does not replace an "
    "agronomist and must not be used to decide access to finance, insurance or public services. "
    "Human review is required."
)


def student_digits(student_id: str) -> str:
    """Return the registration digits found in the username part of the student id.

    Example: 'ug2527104@ines.ac.rw' -> '2527104'.
    """
    username = student_id.split("@")[0]
    digits = re.sub(r"\D", "", username)
    if len(digits) < 4:
        raise ValueError(
            f"Student id '{student_id}' must contain the registration digits (at least 4 digits)."
        )
    return digits


def verification_code(seed: int) -> str:
    """Return the individual verification code AI-A2-LLLL."""
    return f"AI-A2-{seed:04d}"
