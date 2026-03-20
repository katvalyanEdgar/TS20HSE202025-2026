from __future__ import annotations

import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, AutoTheta, Naive, SeasonalNaive


def run_baselines(train_df: pd.DataFrame, horizon: int, seasonality: int) -> dict[str, pd.DataFrame]:
    models = [Naive()]
    if seasonality > 1:
        models.append(SeasonalNaive(season_length=seasonality))
    models.extend(
        [
            AutoETS(season_length=seasonality, model="ZZZ"),
            AutoTheta(season_length=seasonality),
        ]
    )

    forecaster = StatsForecast(models=models, freq=1, n_jobs=-1)
    forecast_df = forecaster.forecast(df=train_df, h=horizon)

    outputs: dict[str, pd.DataFrame] = {}
    for model_name in forecast_df.columns:
        if model_name in {"unique_id", "ds"}:
            continue
        outputs[model_name] = (
            forecast_df[["unique_id", "ds", model_name]]
            .rename(columns={model_name: "y_hat"})
            .sort_values(["unique_id", "ds"])
            .reset_index(drop=True)
        )
    return outputs
