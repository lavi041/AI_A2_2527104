"""KMeans farm clustering (K=2..6, silhouette selection) and plain-language profiles."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline

from src.config import FEATURE_COLUMNS, K_RANGE, NUMERIC_FEATURES
from src.preprocessing import build_preprocessor

NICE = {
    "rainfall_mm": "rainfall",
    "avg_temp_c": "temperature",
    "soil_moisture_pct": "soil moisture",
    "fertilizer_kg": "fertilizer use",
    "field_size_ha": "field size",
    "cooperative_support": "cooperative support",
    "season_A": "share of season A",
}


def _make(k: int, seed: int) -> Pipeline:
    return Pipeline(
        [("pre", build_preprocessor()), ("kmeans", KMeans(n_clusters=k, n_init=10, random_state=seed))]
    )


def run_clustering(train: pd.DataFrame, seed: int):
    """Fit on TRAIN features only (targets are never used). Returns (pipeline, report)."""
    X = train[FEATURE_COLUMNS]
    scores = {}
    for k in K_RANGE:
        pipe = _make(k, seed).fit(X)
        Z = pipe[:-1].transform(X)
        scores[k] = round(float(silhouette_score(Z, pipe["kmeans"].labels_)), 4)
    best_k = max(scores, key=scores.get)
    pipe = _make(best_k, seed).fit(X)
    labels = pipe["kmeans"].labels_
    profiles = build_profiles(train, labels)
    s = scores[best_k]
    strength = (
        "strong" if s >= 0.5 else "reasonable" if s >= 0.35 else "weak" if s >= 0.2 else "very weak"
    )
    report = {
        "silhouette_by_k": {str(k): v for k, v in scores.items()},
        "selected_k": best_k,
        "selected_silhouette": s,
        "structure_strength": strength,
        "profiles": profiles,
        "interpretation": (
            f"K={best_k} gave the highest silhouette ({s}). A score this low ({strength} structure) "
            "means the clusters are practical groupings of similar farms rather than sharply "
            "separated natural segments. Use them to organise field visits, not as hard categories."
        ),
        "note": "Clustering used only the 7 input features; yield and risk are shown post-hoc.",
    }
    return pipe, report


def build_profiles(train: pd.DataFrame, labels: np.ndarray) -> list[dict]:
    """Describe each cluster relative to the overall training average (z-score of means)."""
    df = train[NUMERIC_FEATURES + ["cooperative_support", "yield_tons", "high_risk"]].copy()
    df["season_A"] = (train["season"] == "A").astype(float)
    df["cluster"] = labels
    feats = NUMERIC_FEATURES + ["cooperative_support", "season_A"]
    overall_mean, overall_std = df[feats].mean(), df[feats].std(ddof=0)
    out = []
    for c in sorted(df["cluster"].unique()):
        g = df[df["cluster"] == c]
        z = ((g[feats].mean() - overall_mean) / overall_std).sort_values(key=np.abs, ascending=False)
        strong = [(f, v) for f, v in z.items() if abs(v) >= 0.4][:3]
        desc = (
            ", ".join(f"{'high' if v > 0 else 'low'} {NICE[f]}" for f, v in strong)
            if strong
            else "close to the average on every feature"
        )
        out.append(
            {
                "cluster": int(c),
                "size": int(len(g)),
                "label": f"Cluster {int(c)}: {desc}",
                "feature_means": {f: round(float(g[f].mean()), 3) for f in feats},
                "z_scores": {f: round(float(z[f]), 3) for f in feats},
                "posthoc_mean_yield_tons": round(float(g["yield_tons"].mean()), 3),
                "posthoc_high_risk_share": round(float(g["high_risk"].mean()), 3),
            }
        )
    return out
