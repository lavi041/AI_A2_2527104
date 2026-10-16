"""Seed, schema, range, missing-value, duplicate and leakage checks plus cleaning."""
from __future__ import annotations

import pandas as pd

from src.config import (
    CLASSIFICATION_TARGET,
    FORBIDDEN_FEATURES,
    NUMERIC_RANGES,
    REGRESSION_TARGET,
    REQUIRED_COLUMNS,
    student_digits,
    verification_code,
)


class DataValidationError(ValueError):
    """Raised when identity, schema, range or leakage checks fail."""


def validate_seed(student_id: str, seed: int) -> str:
    """Reject a seed that is not LLLL (last four digits of the registration number).

    Returns the verification code AI-A2-LLLL.
    """
    digits = student_digits(student_id)
    expected = int(digits[-4:])
    if seed != expected:
        raise DataValidationError(
            f"Invalid seed {seed}: for student id '{student_id}' the seed must be {expected} "
            f"(last four digits of the registration number)."
        )
    return verification_code(seed)


def validate_schema(df: pd.DataFrame) -> None:
    """Check that all required columns exist."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(f"Dataset is missing required columns: {missing}")


def validate_ranges(df: pd.DataFrame) -> None:
    """Check value ranges and categories (missing values are allowed here and imputed later)."""
    problems = []
    for col, (lo, hi) in NUMERIC_RANGES.items():
        s = df[col].dropna()
        bad = int(((s < lo) | (s > hi)).sum())
        if bad:
            problems.append(f"{col}: {bad} value(s) outside [{lo}, {hi}]")
    if not df["season"].dropna().isin(["A", "B"]).all():
        problems.append("season must be 'A' or 'B'")
    if not df["cooperative_support"].dropna().isin([0, 1]).all():
        problems.append("cooperative_support must be 0 or 1")
    if not df[CLASSIFICATION_TARGET].dropna().isin([0, 1]).all():
        problems.append("high_risk must be 0 or 1")
    if (df[REGRESSION_TARGET].dropna() < 0.5).any():
        problems.append("yield_tons must be >= 0.50")
    if df[REGRESSION_TARGET].isna().any() or df[CLASSIFICATION_TARGET].isna().any():
        problems.append("targets must not contain missing values")
    if problems:
        raise DataValidationError("Range check failed: " + "; ".join(problems))


def assert_no_leakage(feature_columns: list[str]) -> None:
    """Fail if a target-derived column is used as an input feature."""
    leaked = [c for c in feature_columns if c in FORBIDDEN_FEATURES]
    if leaked:
        raise DataValidationError(f"Target leakage: {leaked} cannot be input features.")


def validate_and_clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Validate the raw dataset, drop exact duplicates and return (clean_df, report).

    Missing feature values are NOT filled here; they are imputed inside the sklearn
    Pipeline so that imputation statistics come from the training split only.
    """
    validate_schema(df)
    validate_ranges(df)
    n_raw = len(df)
    n_dup = int(df.duplicated().sum())
    clean = df.drop_duplicates(keep="first")  # original row index is kept as row_id
    report = {
        "rows_raw": n_raw,
        "duplicate_rows_found": n_dup,
        "duplicate_rows_removed": n_raw - len(clean),
        "rows_after_cleaning": len(clean),
        "missing_values_after_dedup": {
            c: int(v) for c, v in clean.isna().sum().items() if v > 0
        },
        "missing_values_raw": {c: int(v) for c, v in df.isna().sum().items() if v > 0},
        "high_risk_share": round(float(clean[CLASSIFICATION_TARGET].mean()), 4),
        "checks_passed": ["schema", "ranges", "categories", "targets_not_missing"],
    }
    return clean, report
