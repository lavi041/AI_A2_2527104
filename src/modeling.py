"""Regression and classification: baselines, candidates, CV-based selection, test metrics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline

from src.config import CV_FOLDS, FEATURE_COLUMNS
from src.preprocessing import build_preprocessor


def _pipe(estimator) -> Pipeline:
    return Pipeline([("pre", build_preprocessor()), ("model", estimator)])


def regression_candidates(seed: int) -> dict:
    """Mean baseline + three candidate regressors."""
    return {
        "baseline_mean": DummyRegressor(strategy="mean"),
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=1),
        "gradient_boosting": GradientBoostingRegressor(random_state=seed),
    }


def classification_candidates(seed: int) -> dict:
    """Majority baseline + three candidate classifiers."""
    return {
        "baseline_majority": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1),
        "gradient_boosting": GradientBoostingClassifier(random_state=seed),
    }


def _r(x: float) -> float:
    return round(float(x), 4)


def train_regression(train: pd.DataFrame, test: pd.DataFrame, target: str, seed: int):
    """Fit all regressors on train; select by 5-fold CV RMSE; report held-out test metrics."""
    X_tr, y_tr = train[FEATURE_COLUMNS], train[target]
    X_te, y_te = test[FEATURE_COLUMNS], test[target]
    cv = KFold(CV_FOLDS, shuffle=True, random_state=seed)
    results, fitted = {}, {}
    for name, est in regression_candidates(seed).items():
        pipe = _pipe(est)
        cv_rmse = -cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="neg_root_mean_squared_error")
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        results[name] = {
            "cv_rmse_mean": _r(cv_rmse.mean()),
            "test_mae": _r(mean_absolute_error(y_te, pred)),
            "test_rmse": _r(np.sqrt(mean_squared_error(y_te, pred))),
            "test_r2": _r(r2_score(y_te, pred)),
        }
        fitted[name] = pipe
    candidates = [n for n in results if not n.startswith("baseline")]
    best = min(candidates, key=lambda n: results[n]["cv_rmse_mean"])
    base = results["baseline_mean"]["cv_rmse_mean"]
    reason = (
        f"{best} has the lowest 5-fold cross-validated RMSE on the training split "
        f"({results[best]['cv_rmse_mean']} vs {base} for the mean baseline). Selection used "
        f"training-only cross-validation, not the test set."
    )
    pred = fitted[best].predict(X_te)
    resid = y_te.to_numpy() - pred
    worst = int(np.argmax(np.abs(resid)))
    row = test.iloc[worst]
    largest_error = {
        "row_id": int(test.index[worst]),
        "actual_yield_tons": _r(y_te.iloc[worst]),
        "predicted_yield_tons": _r(pred[worst]),
        "residual": _r(resid[worst]),
        "features": {c: (None if pd.isna(row[c]) else (row[c] if isinstance(row[c], str) else _r(row[c]))) for c in FEATURE_COLUMNS},
    }
    return results, fitted, best, reason, pred, largest_error


def train_classification(train: pd.DataFrame, test: pd.DataFrame, target: str, seed: int):
    """Fit all classifiers on train; select by 5-fold CV F1; report held-out test metrics."""
    X_tr, y_tr = train[FEATURE_COLUMNS], train[target]
    X_te, y_te = test[FEATURE_COLUMNS], test[target]
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=seed)
    results, fitted = {}, {}
    for name, est in classification_candidates(seed).items():
        pipe = _pipe(est)
        cv_f1 = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="f1")
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        proba = pipe.predict_proba(X_te)[:, 1]
        try:
            auc = _r(roc_auc_score(y_te, proba)) if y_te.nunique() == 2 else None
        except ValueError:
            auc = None
        results[name] = {
            "cv_f1_mean": _r(cv_f1.mean()),
            "test_precision": _r(precision_score(y_te, pred, zero_division=0)),
            "test_recall": _r(recall_score(y_te, pred, zero_division=0)),
            "test_f1": _r(f1_score(y_te, pred, zero_division=0)),
            "test_roc_auc": auc,
            "confusion_matrix": confusion_matrix(y_te, pred, labels=[0, 1]).tolist(),
        }
        fitted[name] = pipe
    candidates = [n for n in results if not n.startswith("baseline")]
    best = max(candidates, key=lambda n: results[n]["cv_f1_mean"])
    reason = (
        f"{best} has the highest 5-fold cross-validated F1 on the training split "
        f"({results[best]['cv_f1_mean']} vs {results['baseline_majority']['cv_f1_mean']} for the "
        f"majority baseline). Selection used training-only cross-validation, not the test set."
    )
    pred = fitted[best].predict(X_te)
    proba = fitted[best].predict_proba(X_te)[:, 1]
    # Most serious error: a truly high-risk farm that the model missed (false negative).
    fn = np.where((y_te.to_numpy() == 1) & (pred == 0))[0]
    err = None
    if len(fn):
        i = int(fn[np.argmin(proba[fn])])
        row = test.iloc[i]
        err = {
            "type": "false_negative (high-risk farm predicted as not high-risk)",
            "row_id": int(test.index[i]),
            "predicted_probability_high_risk": _r(proba[i]),
            "features": {c: (None if pd.isna(row[c]) else (row[c] if isinstance(row[c], str) else _r(row[c]))) for c in FEATURE_COLUMNS},
            "actual_yield_tons": _r(row["yield_tons"]),
        }
    return results, fitted, best, reason, pred, proba, err
