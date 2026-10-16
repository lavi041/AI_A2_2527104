"""Train-only preprocessing: ColumnTransformer shared by training and inference."""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    BINARY_FEATURES,
    CATEGORICAL_FEATURES,
    CLASSIFICATION_TARGET,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TEST_SIZE,
)
from src.validation import assert_no_leakage


def build_preprocessor() -> ColumnTransformer:
    """Median imputation + scaling for numerics, one-hot for season, passthrough for binary."""
    assert_no_leakage(FEATURE_COLUMNS)
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    return ColumnTransformer(
        [
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", OneHotEncoder(drop="if_binary", handle_unknown="error"), CATEGORICAL_FEATURES),
            ("bin", "passthrough", BINARY_FEATURES),
        ]
    )


def split_data(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fixed train/test split (random_state=LLLL), stratified on high_risk."""
    train, test = train_test_split(
        df, test_size=TEST_SIZE, random_state=seed, stratify=df[CLASSIFICATION_TARGET]
    )
    return train, test
