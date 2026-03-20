from __future__ import annotations

import math

import numpy as np
import pandas as pd


def _mase_scale(values: np.ndarray, seasonality: int) -> float:
    if seasonality > 1 and len(values) > seasonality:
        diffs = np.abs(values[seasonality:] - values[:-seasonality])
    else:
        diffs = np.abs(np.diff(values))
    if len(diffs) == 0:
        return 1.0
    scale = float(np.mean(diffs))
    if scale < 1e-8:
        scale = float(np.mean(np.abs(values)))
    return max(scale, 1.0)


def evaluate_forecasts(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    forecasts: dict[str, pd.DataFrame],
    seasonality: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    actuals = test_df.sort_values(["unique_id", "ds"]).reset_index(drop=True).copy()
    actuals["horizon_step"] = actuals.groupby("unique_id").cumcount() + 1

    scale_df = (
        train_df.groupby("unique_id")["y"]
        .apply(lambda values: _mase_scale(values.to_numpy(dtype=float), seasonality))
        .rename("mase_scale")
        .reset_index()
    )

    overall_rows: list[dict[str, float | str]] = []
    horizon_rows: list[dict[str, float | str | int]] = []
    prediction_frames: list[pd.DataFrame] = []

    for model_name, forecast_df in forecasts.items():
        merged = (
            actuals.merge(
                forecast_df.rename(columns={"y_hat": "prediction"}),
                on=["unique_id", "ds"],
                how="inner",
            )
            .merge(scale_df, on="unique_id", how="left")
            .sort_values(["unique_id", "ds"])
            .reset_index(drop=True)
        )
        merged["error"] = merged["prediction"] - merged["y"]
        merged["abs_error"] = merged["error"].abs()
        merged["squared_error"] = merged["error"] ** 2
        merged["smape"] = (
            200.0
            * merged["abs_error"]
            / (merged["prediction"].abs() + merged["y"].abs()).clip(lower=1e-8)
        )
        merged["mase"] = merged["abs_error"] / merged["mase_scale"]
        merged["model"] = model_name
        prediction_frames.append(merged)

        overall_rows.append(
            {
                "model": model_name,
                "smape": float(merged["smape"].mean()),
                "mase": float(merged["mase"].mean()),
                "rmse": float(math.sqrt(merged["squared_error"].mean())),
            }
        )

        horizon_summary = (
            merged.groupby("horizon_step", as_index=False)
            .agg(
                smape=("smape", "mean"),
                mase=("mase", "mean"),
                rmse=("squared_error", lambda x: math.sqrt(float(np.mean(x)))),
            )
        )
        horizon_summary["model"] = model_name
        horizon_rows.extend(horizon_summary.to_dict("records"))

    metrics_df = pd.DataFrame(overall_rows).sort_values("smape").reset_index(drop=True)
    horizon_df = pd.DataFrame(horizon_rows).sort_values(["model", "horizon_step"]).reset_index(drop=True)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)
    return metrics_df, horizon_df, predictions_df
