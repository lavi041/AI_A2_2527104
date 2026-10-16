"""Load saved artifacts and run one validated prediction."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd

from src.config import FEATURE_COLUMNS, MODELS_DIR
from src.schemas import FarmRecord


class ModelsNotFoundError(FileNotFoundError):
    """Raised when saved models are missing (run run_all.py first)."""


@lru_cache(maxsize=2)
def load_bundle(models_dir: str = str(MODELS_DIR)) -> dict:
    """Load regression, classification and clustering pipelines plus metadata."""
    d = Path(models_dir)
    needed = ["regression_model.joblib", "classification_model.joblib",
              "clustering_model.joblib", "metadata.json"]
    missing = [n for n in needed if not (d / n).exists()]
    if missing:
        raise ModelsNotFoundError(f"Missing model files {missing} in {d}. Run: python run_all.py --student-id <id> --seed <LLLL>")
    return {
        "reg": joblib.load(d / "regression_model.joblib"),
        "clf": joblib.load(d / "classification_model.joblib"),
        "clu": joblib.load(d / "clustering_model.joblib"),
        "meta": json.loads((d / "metadata.json").read_text(encoding="utf-8")),
    }


def predict_record(record: dict | FarmRecord, bundle: dict | None = None) -> dict:
    """Validate (pydantic raises ValidationError) and predict for one farm record."""
    rec = record if isinstance(record, FarmRecord) else FarmRecord.model_validate(record)
    bundle = bundle or load_bundle()
    X = pd.DataFrame([rec.model_dump()])[FEATURE_COLUMNS]
    yield_est = float(bundle["reg"].predict(X)[0])
    proba = float(bundle["clf"].predict_proba(X)[0, 1])
    pred = int(proba >= 0.5)
    cluster = int(bundle["clu"].predict(X)[0])
    profile = next(p["label"] for p in bundle["meta"]["clustering"]["profiles"] if p["cluster"] == cluster)
    return {
        "yield_estimate_tons": round(yield_est, 4),
        "high_risk_prediction": pred,
        "high_risk_label": "high risk" if pred else "not high risk",
        "high_risk_probability": round(proba, 4),
        "cluster": cluster,
        "cluster_profile": profile,
        "verification_code": bundle["meta"]["verification_code"],
        "disclaimer": bundle["meta"]["disclaimer"],
    }
