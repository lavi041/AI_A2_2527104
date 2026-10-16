"""Automated tests: happy paths AND failure paths for the Musanze Harvest Intelligence Service."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from data.generate_data import generate_dataframe, sha256_of_file, write_csv
from src.config import FEATURE_COLUMNS, MODELS_DIR, PROJECT_ROOT
from src.inference import load_bundle, predict_record
from src.pipeline import run_pipeline
from src.validation import DataValidationError, assert_no_leakage, validate_seed

STUDENT_ID = "ug2527104@ines.ac.rw"
SEED = 7104
VALID = {
    "rainfall_mm": 1050.0, "avg_temp_c": 18.5, "soil_moisture_pct": 55.0,
    "fertilizer_kg": 120.0, "field_size_ha": 1.8, "season": "A", "cooperative_support": 1,
}


@pytest.fixture(scope="session", autouse=True)
def trained_models():
    """Make sure saved models exist (run the pipeline once if they do not)."""
    if not (MODELS_DIR / "metadata.json").exists():
        run_pipeline(STUDENT_ID, SEED, verbose=False)
    load_bundle.cache_clear()
    return load_bundle()


@pytest.fixture()
def client():
    from app import app
    return TestClient(app)


def test_valid_inference_returns_all_fields():
    out = predict_record(VALID)
    assert out["verification_code"] == "AI-A2-7104"
    assert out["yield_estimate_tons"] > 0
    assert out["high_risk_prediction"] in (0, 1)
    assert 0.0 <= out["high_risk_probability"] <= 1.0
    assert isinstance(out["cluster"], int)


def test_missing_field_is_rejected():
    bad = {k: v for k, v in VALID.items() if k != "field_size_ha"}
    with pytest.raises(ValidationError) as exc:
        predict_record(bad)
    assert any(e["type"] == "missing" for e in exc.value.errors())


def test_wrong_type_is_rejected():
    with pytest.raises(ValidationError):
        predict_record({**VALID, "rainfall_mm": "lots"})


def test_out_of_range_value_is_rejected():
    with pytest.raises(ValidationError):
        predict_record({**VALID, "soil_moisture_pct": 150})


def test_unknown_category_and_extra_field_are_rejected():
    with pytest.raises(ValidationError):
        predict_record({**VALID, "season": "C"})
    with pytest.raises(ValidationError):
        predict_record({**VALID, "yield_tons": 9.0})  # target must never be accepted as input


def test_reproducibility_same_fingerprint_and_metrics(tmp_path):
    """Two independent runs give identical fingerprints and metrics (and match the submitted CSV)."""
    r1 = run_pipeline(STUDENT_ID, SEED, root=tmp_path / "a", verbose=False)
    r2 = run_pipeline(STUDENT_ID, SEED, root=tmp_path / "b", verbose=False)
    assert r1["sha256"] == r2["sha256"]
    assert r1["regression"] == r2["regression"]
    assert r1["classification"] == r2["classification"]
    assert r1["clustering"]["silhouette_by_k"] == r2["clustering"]["silhouette_by_k"]
    submitted = PROJECT_ROOT / "data" / "AI_A2_2527104.csv"
    if submitted.exists():
        assert sha256_of_file(submitted) == r1["sha256"]


def test_dataset_signature_rows_missing_and_duplicates(tmp_path):
    df = generate_dataframe(SEED)
    base = 240 + SEED % 41                      # 251 base rows for LLLL = 7104
    m = round(0.05 * base)                      # 13 deliberate missing values per column
    assert len(df) == base + 3 == 254
    assert df.duplicated().sum() == 3
    assert df.iloc[:base]["soil_moisture_pct"].isna().sum() == m
    assert df.iloc[:base]["fertilizer_kg"].isna().sum() == m
    write_csv(df, tmp_path / "a.csv")
    write_csv(df, tmp_path / "b.csv")
    assert (tmp_path / "a.csv").read_bytes() == (tmp_path / "b.csv").read_bytes()


def test_wrong_seed_is_rejected():
    assert validate_seed(STUDENT_ID, SEED) == "AI-A2-7104"
    with pytest.raises(DataValidationError):
        validate_seed(STUDENT_ID, 1234)


def test_no_target_leakage_in_features():
    assert not {"yield_tons", "high_risk", "yield_per_ha"} & set(FEATURE_COLUMNS)
    with pytest.raises(DataValidationError):
        assert_no_leakage(FEATURE_COLUMNS + ["high_risk"])


def test_api_health_model_info_and_predict(client):
    assert client.get("/health").json()["status"] == "ok"
    info = client.get("/model-info").json()
    assert info["verification_code"] == "AI-A2-7104" and info["synthetic_data"] is True
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200 and r.json()["verification_code"] == "AI-A2-7104"


def test_api_invalid_requests_return_422_without_echoing_values(client):
    r = client.post("/predict", json={**VALID, "rainfall_mm": 99999})
    assert r.status_code == 422 and r.json()["error"] == "validation_error"
    assert "99999" not in r.text
    assert client.post("/predict", json={"season": "A"}).status_code == 422
    assert client.post("/predict", content="not json", headers={"content-type": "application/json"}).status_code == 422
