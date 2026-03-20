from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .baselines import run_baselines
from .clustering import compute_cluster_artifacts
from .config import ExperimentConfig
from .data import load_m4_sample
from .evaluation import evaluate_forecasts
from .models import run_catboost_modes


def _ensure_output_dirs(config: ExperimentConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.plots_dir.mkdir(parents=True, exist_ok=True)


def _plot_metrics(metrics_df: pd.DataFrame, path: str) -> None:
    ordered = metrics_df.sort_values("smape")
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.bar(ordered["model"], ordered["smape"], color="#3b6fb6")
    axis.set_title("Итоговый sMAPE по моделям")
    axis.set_ylabel("sMAPE")
    axis.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_horizon_metrics(horizon_df: pd.DataFrame, path: str) -> None:
    fig, axis = plt.subplots(figsize=(10, 5))
    for model_name, group in horizon_df.groupby("model", sort=False):
        axis.plot(group["horizon_step"], group["smape"], marker="o", linewidth=1.8, label=model_name)
    axis.set_title("sMAPE по шагам горизонта")
    axis.set_xlabel("Шаг прогноза")
    axis.set_ylabel("sMAPE")
    axis.legend(ncols=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_cluster_examples(train_df: pd.DataFrame, assignments_df: pd.DataFrame, path: str, seed: int) -> None:
    merged = train_df.merge(assignments_df, on="unique_id", how="left")
    clusters = sorted(merged["cluster"].dropna().unique())
    fig, axes = plt.subplots(len(clusters), 1, figsize=(10, max(4, 3 * len(clusters))), squeeze=False)

    for axis, cluster_id in zip(axes.flatten(), clusters):
        cluster_df = merged[merged["cluster"] == cluster_id]
        sample_ids = (
            cluster_df["unique_id"]
            .drop_duplicates()
            .sample(n=min(4, cluster_df["unique_id"].nunique()), random_state=seed)
            .tolist()
        )
        for unique_id in sample_ids:
            series = cluster_df[cluster_df["unique_id"] == unique_id].sort_values("ds")["y"].to_numpy(dtype=float)
            standardized = (series - np.mean(series)) / max(np.std(series), 1e-8)
            axis.plot(standardized, linewidth=1.2, alpha=0.85, label=unique_id)
        axis.set_title(f"Кластер {cluster_id}")
        axis.legend(fontsize=7, ncols=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_experiment(config: ExperimentConfig) -> dict[str, str | int | float]:
    _ensure_output_dirs(config)

    data_bundle = load_m4_sample(config)
    cluster_artifacts = compute_cluster_artifacts(
        train_df=data_bundle.train_df,
        seasonality=data_bundle.seasonality,
        max_clusters=config.max_clusters,
        random_seed=config.random_seed,
    )
    static_with_clusters = data_bundle.static_df.merge(
        cluster_artifacts.assignments_df, on="unique_id", how="left"
    )

    forecasts = run_baselines(
        train_df=data_bundle.train_df,
        horizon=data_bundle.horizon,
        seasonality=data_bundle.seasonality,
    )
    catboost_artifacts = run_catboost_modes(
        train_df=data_bundle.train_df,
        test_df=data_bundle.test_df,
        static_df=static_with_clusters,
        config=config,
        seasonality=data_bundle.seasonality,
    )
    forecasts.update(catboost_artifacts.forecasts)

    metrics_df, horizon_df, predictions_df = evaluate_forecasts(
        train_df=data_bundle.train_df,
        test_df=data_bundle.test_df,
        forecasts=forecasts,
        seasonality=data_bundle.seasonality,
    )

    metrics_df.to_csv(config.output_dir / "metrics.csv", index=False)
    horizon_df.to_csv(config.output_dir / "horizon_metrics.csv", index=False)
    predictions_df.to_csv(config.output_dir / "predictions.csv", index=False)
    cluster_artifacts.assignments_df.to_csv(config.output_dir / "cluster_assignments.csv", index=False)
    cluster_artifacts.summary_df.to_csv(config.output_dir / "cluster_feature_summary.csv", index=False)
    cluster_artifacts.silhouette_df.to_csv(config.output_dir / "silhouette_scores.csv", index=False)
    catboost_artifacts.scales_df.to_csv(config.output_dir / "series_scales.csv", index=False)

    _plot_metrics(metrics_df, str(config.plots_dir / "metrics_smape.png"))
    _plot_horizon_metrics(horizon_df, str(config.plots_dir / "horizon_smape.png"))
    _plot_cluster_examples(
        train_df=data_bundle.train_df,
        assignments_df=cluster_artifacts.assignments_df,
        path=str(config.plots_dir / "cluster_examples.png"),
        seed=config.random_seed,
    )

    best_row = metrics_df.iloc[0]
    summary = {
        "группа": config.group,
        "число_рядов": int(data_bundle.train_df["unique_id"].nunique()),
        "горизонт": int(data_bundle.horizon),
        "сезонность": int(data_bundle.seasonality),
        "лучшая_модель": str(best_row["model"]),
        "лучший_smape": float(best_row["smape"]),
    }
    (config.output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary
