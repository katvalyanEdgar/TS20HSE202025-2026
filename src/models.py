from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from .config import ExperimentConfig


@dataclass(slots=True)
class CatBoostArtifacts:
    forecasts: dict[str, pd.DataFrame]
    scales_df: pd.DataFrame


def _compute_scale_map(train_df: pd.DataFrame, use_mean_scaling: bool) -> dict[str, float]:
    scale_map: dict[str, float] = {}
    for unique_id, group in train_df.groupby("unique_id", sort=False):
        values = group["y"].to_numpy(dtype=float)
        if not use_mean_scaling:
            scale_map[unique_id] = 1.0
            continue
        scale = float(np.mean(np.abs(values)))
        if scale < 1e-8:
            scale = float(np.max(np.abs(values)))
        if scale < 1e-8:
            scale = 1.0
        scale_map[unique_id] = scale
    return scale_map


def _base_feature_row(
    history: np.ndarray,
    next_ds: int,
    seasonality: int,
    lags: Iterable[int],
    rolling_windows: Iterable[int],
) -> dict[str, float]:
    row: dict[str, float] = {}
    rolling_windows = tuple(rolling_windows)

    for lag in lags:
        row[f"lag_{lag}"] = float(history[-lag])
    for window in rolling_windows:
        row[f"rolling_mean_{window}"] = float(np.mean(history[-window:]))
        row[f"rolling_std_{window}"] = float(np.std(history[-window:]))

    recent_window = min(len(history), max(rolling_windows))
    row["diff_1"] = float(history[-1] - history[-2])
    row["recent_mean"] = float(np.mean(history[-recent_window:]))
    row["recent_std"] = float(np.std(history[-recent_window:]))
    row["time_idx"] = float(len(history) + 1)

    if seasonality > 1:
        seasonal_position = (next_ds - 1) % seasonality
        angle = 2.0 * np.pi * seasonal_position / seasonality
        row["season_sin"] = float(np.sin(angle))
        row["season_cos"] = float(np.cos(angle))
        row["seasonal_diff"] = float(history[-1] - history[-seasonality])
    else:
        row["season_sin"] = 0.0
        row["season_cos"] = 1.0
        row["seasonal_diff"] = 0.0
    return row


def _build_supervised_frame(
    train_df: pd.DataFrame,
    static_df: pd.DataFrame,
    scale_map: dict[str, float],
    config: ExperimentConfig,
    seasonality: int,
    *,
    include_id: bool,
    include_category: bool,
    include_cluster: bool,
) -> pd.DataFrame:
    static_lookup = static_df.set_index("unique_id").to_dict("index")
    rows: list[dict[str, float | str]] = []

    for unique_id, group in train_df.groupby("unique_id", sort=False):
        group = group.sort_values("ds")
        history = group["y"].to_numpy(dtype=float) / scale_map[unique_id]
        ds_values = group["ds"].to_numpy(dtype=int)
        for index in range(config.max_lag, len(history)):
            row = _base_feature_row(
                history=history[:index],
                next_ds=int(ds_values[index]),
                seasonality=seasonality,
                lags=config.lags,
                rolling_windows=config.rolling_windows,
            )
            if include_id:
                row["unique_id"] = unique_id
            if include_category:
                row["category"] = str(static_lookup[unique_id]["category"])
            if include_cluster:
                row["cluster"] = f"cluster_{static_lookup[unique_id]['cluster']}"
            row["target"] = float(history[index])
            rows.append(row)

    if not rows:
        raise ValueError("Не удалось сформировать обучающие примеры. Проверьте лаги и длину рядов.")
    return pd.DataFrame(rows)


def _fit_regressor(
    frame: pd.DataFrame,
    config: ExperimentConfig,
    cat_features: list[str],
) -> CatBoostRegressor:
    feature_cols = [column for column in frame.columns if column != "target"]
    model = CatBoostRegressor(
        iterations=config.catboost_iterations,
        depth=config.catboost_depth,
        learning_rate=config.catboost_learning_rate,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=config.random_seed,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(frame[feature_cols], frame["target"], cat_features=cat_features)
    return model


def _forecast_series(
    model: CatBoostRegressor,
    series_df: pd.DataFrame,
    static_row: dict[str, str | int],
    scale: float,
    horizon: int,
    seasonality: int,
    config: ExperimentConfig,
    *,
    include_id: bool,
    include_category: bool,
    include_cluster: bool,
) -> list[dict[str, float | str | int]]:
    group = series_df.sort_values("ds")
    history = (group["y"].to_numpy(dtype=float) / scale).tolist()
    last_ds = int(group["ds"].iloc[-1])
    rows: list[dict[str, float | str | int]] = []

    for step in range(1, horizon + 1):
        feature_row = _base_feature_row(
            history=np.asarray(history, dtype=float),
            next_ds=last_ds + step,
            seasonality=seasonality,
            lags=config.lags,
            rolling_windows=config.rolling_windows,
        )
        if include_id:
            feature_row["unique_id"] = str(group["unique_id"].iloc[0])
        if include_category:
            feature_row["category"] = str(static_row["category"])
        if include_cluster:
            feature_row["cluster"] = f"cluster_{static_row['cluster']}"

        prediction = float(model.predict(pd.DataFrame([feature_row]))[0])
        history.append(prediction)
        rows.append(
            {
                "unique_id": str(group["unique_id"].iloc[0]),
                "ds": last_ds + step,
                "y_hat": prediction * scale,
            }
        )
    return rows


def run_catboost_modes(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    static_df: pd.DataFrame,
    config: ExperimentConfig,
    seasonality: int,
) -> CatBoostArtifacts:
    static_df = static_df.copy()
    static_lookup = static_df.set_index("unique_id").to_dict("index")
    scale_map = _compute_scale_map(train_df, config.use_mean_scaling)
    scale_rows = [{"unique_id": unique_id, "scale": scale} for unique_id, scale in scale_map.items()]

    forecasts: dict[str, pd.DataFrame] = {}
    horizon = int(test_df.groupby("unique_id")["ds"].size().iloc[0])

    global_frame = _build_supervised_frame(
        train_df=train_df,
        static_df=static_df,
        scale_map=scale_map,
        config=config,
        seasonality=seasonality,
        include_id=True,
        include_category=True,
        include_cluster=True,
    )
    global_model = _fit_regressor(global_frame, config, ["unique_id", "category", "cluster"])
    global_rows: list[dict[str, float | str | int]] = []
    for unique_id, series_df in train_df.groupby("unique_id", sort=False):
        global_rows.extend(
            _forecast_series(
                model=global_model,
                series_df=series_df,
                static_row=static_lookup[unique_id],
                scale=scale_map[unique_id],
                horizon=horizon,
                seasonality=seasonality,
                config=config,
                include_id=True,
                include_category=True,
                include_cluster=True,
            )
        )
    forecasts["CatBoostGlobal"] = (
        pd.DataFrame(global_rows).sort_values(["unique_id", "ds"]).reset_index(drop=True)
    )

    cluster_rows: list[dict[str, float | str | int]] = []
    for cluster_id, cluster_static in static_df.groupby("cluster", sort=True):
        cluster_ids = cluster_static["unique_id"].tolist()
        cluster_train = train_df[train_df["unique_id"].isin(cluster_ids)].copy()
        cluster_frame = _build_supervised_frame(
            train_df=cluster_train,
            static_df=cluster_static,
            scale_map=scale_map,
            config=config,
            seasonality=seasonality,
            include_id=True,
            include_category=True,
            include_cluster=False,
        )
        cluster_model = _fit_regressor(cluster_frame, config, ["unique_id", "category"])
        for unique_id, series_df in cluster_train.groupby("unique_id", sort=False):
            cluster_rows.extend(
                _forecast_series(
                    model=cluster_model,
                    series_df=series_df,
                    static_row=static_lookup[unique_id],
                    scale=scale_map[unique_id],
                    horizon=horizon,
                    seasonality=seasonality,
                    config=config,
                    include_id=True,
                    include_category=True,
                    include_cluster=False,
                )
            )
    forecasts["CatBoostClusterGlobal"] = (
        pd.DataFrame(cluster_rows).sort_values(["unique_id", "ds"]).reset_index(drop=True)
    )

    local_rows: list[dict[str, float | str | int]] = []
    local_config = ExperimentConfig(
        group=config.group,
        n_series=config.n_series,
        max_clusters=config.max_clusters,
        random_seed=config.random_seed,
        lags=config.lags,
        rolling_windows=config.rolling_windows,
        catboost_iterations=min(config.catboost_iterations, 250),
        catboost_depth=max(4, config.catboost_depth - 1),
        catboost_learning_rate=config.catboost_learning_rate,
        use_mean_scaling=config.use_mean_scaling,
        data_dir=config.data_dir,
        output_dir=config.output_dir,
    )
    for unique_id, series_df in train_df.groupby("unique_id", sort=False):
        local_static = static_df[static_df["unique_id"] == unique_id].copy()
        local_frame = _build_supervised_frame(
            train_df=series_df.copy(),
            static_df=local_static,
            scale_map=scale_map,
            config=local_config,
            seasonality=seasonality,
            include_id=False,
            include_category=False,
            include_cluster=False,
        )
        local_model = _fit_regressor(local_frame, local_config, [])
        local_rows.extend(
            _forecast_series(
                model=local_model,
                series_df=series_df,
                static_row=static_lookup[unique_id],
                scale=scale_map[unique_id],
                horizon=horizon,
                seasonality=seasonality,
                config=local_config,
                include_id=False,
                include_category=False,
                include_cluster=False,
            )
        )
    forecasts["CatBoostLocal"] = (
        pd.DataFrame(local_rows).sort_values(["unique_id", "ds"]).reset_index(drop=True)
    )

    return CatBoostArtifacts(forecasts=forecasts, scales_df=pd.DataFrame(scale_rows))
