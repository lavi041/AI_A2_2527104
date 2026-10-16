"""End-to-end pipeline: data -> validation -> split -> 3 ML tasks -> artifacts + saved models."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from data.generate_data import generate_dataframe, write_csv, write_fingerprint
from src import evaluation
from src.clustering import run_clustering
from src.config import (
    CLASSIFICATION_TARGET,
    DISCLAIMER,
    FEATURE_COLUMNS,
    PROJECT_ROOT,
    REGRESSION_TARGET,
    student_digits,
)
from src.modeling import train_classification, train_regression
from src.preprocessing import split_data
from src.validation import assert_no_leakage, validate_and_clean, validate_seed


def _dump(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_pipeline(student_id: str, seed: int, root: Path = PROJECT_ROOT, verbose: bool = True) -> dict:
    """Run everything and regenerate every artifact under `root`. Returns a summary dict."""
    code = validate_seed(student_id, seed)
    regdigits = student_digits(student_id)
    data_dir, models_dir = root / "data", root / "models"
    art_dir, plots_dir = root / "artifacts", root / "artifacts" / "plots"
    for d in (data_dir, models_dir, art_dir, plots_dir):
        d.mkdir(parents=True, exist_ok=True)
    say = print if verbose else (lambda *a, **k: None)

    say(f"[{code}] student={student_id} seed={seed}")
    # 1. data + fingerprint
    csv_path = data_dir / f"AI_A2_{regdigits}.csv"
    raw = generate_dataframe(seed)
    write_csv(raw, csv_path)
    digest = write_fingerprint(csv_path, seed, len(raw), art_dir / "data_fingerprint.txt")
    say(f"[{code}] dataset: {csv_path.name}, {len(raw)} rows, SHA-256 {digest}")

    # 2. validation + cleaning (reads the CSV back, as an assessor would)
    df = pd.read_csv(csv_path)
    clean, quality = validate_and_clean(df)
    assert_no_leakage(FEATURE_COLUMNS)
    quality.update({"sha256": digest, "verification_code": code, "seed": seed})
    _dump(quality, art_dir / "data_quality_report.json")
    say(f"[{code}] cleaning: {quality['duplicate_rows_removed']} duplicates removed, "
        f"missing={quality['missing_values_raw']}, rows kept={len(clean)}")

    # 3. split (fixed, random_state = LLLL); every transformer is fit inside pipelines on train only
    train, test = split_data(clean, seed)
    say(f"[{code}] split: train={len(train)} test={len(test)} (random_state={seed})")

    # 4. regression
    r_res, r_fit, r_best, r_reason, r_pred, r_err = train_regression(train, test, REGRESSION_TARGET, seed)
    _dump({"verification_code": code, "seed": seed, "target": REGRESSION_TARGET,
           "train_rows": len(train), "test_rows": len(test), "models": r_res,
           "selected_model": r_best, "selection_reason": r_reason, "largest_error": r_err},
          art_dir / "regression_metrics.json")

    # 5. classification
    c_res, c_fit, c_best, c_reason, c_pred, c_proba, c_err = train_classification(
        train, test, CLASSIFICATION_TARGET, seed)
    _dump({"verification_code": code, "seed": seed, "target": CLASSIFICATION_TARGET,
           "positive_class": "high_risk = 1 (yield per hectare below median)",
           "confusion_matrix_layout": "[[TN, FP], [FN, TP]]",
           "models": c_res, "selected_model": c_best, "selection_reason": c_reason,
           "important_error": c_err}, art_dir / "classification_metrics.json")

    # 6. clustering
    clu_pipe, clu_report = run_clustering(train, seed)
    clu_report.update({"verification_code": code, "seed": seed})
    _dump(clu_report, art_dir / "clustering_report.json")
    test_clusters = clu_pipe.predict(test[FEATURE_COLUMNS])

    # 7. test predictions + plots
    evaluation.save_test_predictions(test, r_pred, c_pred, c_proba, test_clusters,
                                     art_dir / "test_predictions.csv")
    evaluation.plot_residuals(test[REGRESSION_TARGET].to_numpy(), r_pred, code, plots_dir / "residuals.png")
    evaluation.plot_confusion(c_res[c_best]["confusion_matrix"], c_best, code, plots_dir / "confusion_matrix.png")
    evaluation.plot_clusters(clu_pipe, train, test, code, plots_dir / "clusters.png")

    # 8. save models (pipelines include the fitted preprocessing) + metadata
    joblib.dump(r_fit[r_best], models_dir / "regression_model.joblib")
    joblib.dump(c_fit[c_best], models_dir / "classification_model.joblib")
    joblib.dump(clu_pipe, models_dir / "clustering_model.joblib")
    joblib.dump(r_fit[r_best][:-1], models_dir / "preprocessor.joblib")
    metadata = {
        "student_id": student_id, "seed": seed, "verification_code": code,
        "dataset_sha256": digest, "dataset_file": csv_path.name,
        "rows_raw": len(raw), "rows_clean": len(clean), "train_rows": len(train), "test_rows": len(test),
        "features": FEATURE_COLUMNS, "synthetic_data": True,
        "regression": {"selected": r_best, **r_res[r_best]},
        "classification": {"selected": c_best, **{k: v for k, v in c_res[c_best].items()}},
        "clustering": {"selected_k": clu_report["selected_k"],
                       "silhouette": clu_report["selected_silhouette"],
                       "profiles": [{"cluster": p["cluster"], "label": p["label"], "size": p["size"]}
                                    for p in clu_report["profiles"]]},
        "disclaimer": DISCLAIMER,
    }
    _dump(metadata, models_dir / "metadata.json")

    say(f"[{code}] regression : best={r_best} {r_res[r_best]}")
    say(f"[{code}] classifier : best={c_best} {c_res[c_best]}")
    say(f"[{code}] clustering : K={clu_report['selected_k']} silhouette={clu_report['selected_silhouette']}")
    files = sorted(str(p.relative_to(root)) for d in (art_dir, models_dir) for p in d.rglob("*") if p.is_file())
    say(f"[{code}] artifacts generated:")
    for f in files:
        say(f"   - {f}")
    say(f"[{code}] DONE. Dataset fingerprint: {digest}")
    return {"metadata": metadata, "regression": r_res, "classification": c_res,
            "clustering": clu_report, "sha256": digest}
