from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import STL


@dataclass(slots=True)
class ClusterArtifacts:
    assignments_df: pd.DataFrame
    features_df: pd.DataFrame
    summary_df: pd.DataFrame
    silhouette_df: pd.DataFrame


def _safe_autocorr(values: np.ndarray, lag: int) -> float:
    if lag <= 0 or len(values) <= lag:
        return 0.0
    left = values[:-lag]
    right = values[lag:]
    if np.std(left) < 1e-8 or np.std(right) < 1e-8:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _stl_strength(values: np.ndarray, seasonality: int) -> tuple[float, float]:
    if seasonality <= 1 or len(values) < seasonality * 2:
        return 0.0, 0.0
    try:
        decomposition = STL(values, period=seasonality, robust=True).fit()
    except ValueError:
        return 0.0, 0.0
    resid_var = np.var(decomposition.resid)
    trend_strength = max(
        0.0,
        1.0 - resid_var / max(np.var(decomposition.trend + decomposition.resid), 1e-8),
    )
    seasonal_strength = max(
        0.0,
        1.0 - resid_var / max(np.var(decomposition.seasonal + decomposition.resid), 1e-8),
    )
    return float(trend_strength), float(seasonal_strength)


def compute_cluster_artifacts(
    train_df: pd.DataFrame,
    seasonality: int,
    max_clusters: int,
    random_seed: int,
) -> ClusterArtifacts:
    feature_rows: list[dict[str, float | str]] = []
    for unique_id, group in train_df.groupby("unique_id", sort=True):
        values = group["y"].to_numpy(dtype=float)
        x_axis = np.arange(len(values), dtype=float)
        slope = float(np.polyfit(x_axis, values, deg=1)[0]) if len(values) > 1 else 0.0
        mean_value = float(np.mean(values))
        std_value = float(np.std(values))
        trend_strength, seasonal_strength = _stl_strength(values, seasonality)
        feature_rows.append(
            {
                "unique_id": unique_id,
                "length": float(len(values)),
                "mean": mean_value,
                "std": std_value,
                "cv": float(std_value / max(abs(mean_value), 1e-8)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "last": float(values[-1]),
                "slope": slope,
                "acf1": _safe_autocorr(values, 1),
                "seasonal_acf": _safe_autocorr(values, seasonality),
                "trend_strength": trend_strength,
                "seasonal_strength": seasonal_strength,
            }
        )

    features_df = pd.DataFrame(feature_rows)
    feature_cols = [column for column in features_df.columns if column != "unique_id"]
    feature_matrix = StandardScaler().fit_transform(features_df[feature_cols].fillna(0.0))

    upper_k = min(max_clusters, len(features_df) - 1)
    if upper_k < 2:
        assignments_df = features_df[["unique_id"]].assign(cluster=0)
        silhouette_df = pd.DataFrame({"k": [1], "silhouette": [np.nan]})
        summary_df = (
            assignments_df.merge(features_df, on="unique_id", how="left")
            .groupby("cluster", as_index=False)
            .agg(count=("unique_id", "size"), **{column: (column, "mean") for column in feature_cols})
        )
        return ClusterArtifacts(assignments_df, features_df, summary_df, silhouette_df)

    silhouette_rows: list[dict[str, float]] = []
    best_score = -np.inf
    best_labels: np.ndarray | None = None
    best_k = 2
    for k in range(2, upper_k + 1):
        model = KMeans(n_clusters=k, n_init=20, random_state=random_seed)
        labels = model.fit_predict(feature_matrix)
        score = silhouette_score(feature_matrix, labels)
        silhouette_rows.append({"k": float(k), "silhouette": float(score)})
        if score > best_score:
            best_score = score
            best_labels = labels
            best_k = k

    if best_labels is None:
        raise RuntimeError("Не удалось получить метки кластеров.")

    assignments_df = features_df[["unique_id"]].assign(cluster=best_labels.astype(int))
    silhouette_df = pd.DataFrame(silhouette_rows)
    summary_df = (
        assignments_df.merge(features_df, on="unique_id", how="left")
        .groupby("cluster", as_index=False)
        .agg(count=("unique_id", "size"), **{column: (column, "mean") for column in feature_cols})
        .sort_values("cluster")
        .reset_index(drop=True)
    )
    summary_df["selected_k"] = best_k
    return ClusterArtifacts(assignments_df, features_df, summary_df, silhouette_df)
