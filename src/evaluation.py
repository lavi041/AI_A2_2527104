"""Plots and test-prediction table."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402


def save_test_predictions(test, reg_pred, clf_pred, clf_proba, clusters, path) -> pd.DataFrame:
    """Write artifacts/test_predictions.csv."""
    df = pd.DataFrame(
        {
            "row_id": test.index.to_numpy(),
            "actual_yield_tons": test["yield_tons"].to_numpy(),
            "predicted_yield_tons": reg_pred,
            "residual": test["yield_tons"].to_numpy() - reg_pred,
            "abs_error": np.abs(test["yield_tons"].to_numpy() - reg_pred),
            "actual_high_risk": test["high_risk"].to_numpy(),
            "predicted_high_risk": clf_pred,
            "high_risk_probability": clf_proba,
            "misclassified": (test["high_risk"].to_numpy() != clf_pred).astype(int),
            "cluster": clusters,
        }
    )
    df.to_csv(path, index=False, float_format="%.4f", lineterminator="\n")
    return df


def plot_residuals(actual, pred, code, path) -> None:
    resid = actual - pred
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].scatter(pred, resid, s=18, alpha=0.8)
    ax[0].axhline(0, color="red", lw=1)
    ax[0].set(xlabel="Predicted yield (tons)", ylabel="Residual (actual - predicted)", title="Residuals vs prediction")
    ax[1].hist(resid, bins=15, edgecolor="black")
    ax[1].set(xlabel="Residual (tons)", ylabel="Farms", title="Residual distribution")
    fig.suptitle(f"Regression residuals on test set [{code}]")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_confusion(cm, model_name, code, path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i][j], ha="center", va="center", fontsize=16,
                    color="white" if cm[i][j] > np.max(cm) / 2 else "black")
    ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["not high-risk", "high-risk"],
           yticklabels=["not high-risk", "high-risk"], xlabel="Predicted", ylabel="Actual",
           title=f"Confusion matrix: {model_name}\n[{code}]")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_clusters(pipe, train, test, code, path) -> None:
    """2-D PCA view of the scaled features, coloured by cluster."""
    from src.config import FEATURE_COLUMNS

    pre = pipe[:-1]
    Ztr, Zte = pre.transform(train[FEATURE_COLUMNS]), pre.transform(test[FEATURE_COLUMNS])
    pca = PCA(n_components=2, random_state=0).fit(Ztr)
    P_tr, P_te = pca.transform(Ztr), pca.transform(Zte)
    centers = pca.transform(pipe["kmeans"].cluster_centers_)
    lab_tr, lab_te = pipe["kmeans"].labels_, pipe["kmeans"].predict(Zte)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    cmap = plt.get_cmap("tab10")
    for c in range(pipe["kmeans"].n_clusters):
        ax.scatter(*P_tr[lab_tr == c].T, s=22, alpha=0.7, color=cmap(c), label=f"Cluster {c}")
        ax.scatter(*P_te[lab_te == c].T, s=45, marker="^", color=cmap(c), edgecolor="black", linewidth=0.7)
    ax.scatter(*centers.T, marker="X", s=140, color="black", label="Centers")
    ax.set(xlabel=f"PC1 ({pca.explained_variance_ratio_[0]:.0%})", ylabel=f"PC2 ({pca.explained_variance_ratio_[1]:.0%})",
           title=f"Farm clusters (triangles = test farms) [{code}]")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
