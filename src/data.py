from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from datasetsforecast.m4 import M4, M4Info

from .config import ExperimentConfig


@dataclass(slots=True)
class DatasetBundle:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    static_df: pd.DataFrame
    horizon: int
    seasonality: int
    frequency: str


def load_m4_sample(config: ExperimentConfig) -> DatasetBundle:
    full_df, _, static_df = M4.load(directory=str(config.data_dir), group=config.group)
    meta = M4Info[config.group]
    full_df = full_df.copy()
    full_df["ds"] = full_df["ds"].astype(int)
    static_df = static_df.copy()
    static_df["category"] = static_df["category"].astype(str)

    series_lengths = full_df.groupby("unique_id")["ds"].size().rename("series_length")
    eligible_ids = series_lengths[series_lengths >= config.min_required_length(meta.horizon)].index
    if len(eligible_ids) == 0:
        raise ValueError("После фильтрации по минимальной длине не осталось подходящих рядов.")

    sampled_ids = (
        series_lengths.loc[eligible_ids]
        .sample(n=min(config.n_series, len(eligible_ids)), random_state=config.random_seed)
        .index
        .tolist()
    )

    sampled_df = (
        full_df[full_df["unique_id"].isin(sampled_ids)]
        .sort_values(["unique_id", "ds"])
        .reset_index(drop=True)
    )
    sampled_df["series_length"] = sampled_df.groupby("unique_id")["ds"].transform("size")
    sampled_df["row_number"] = sampled_df.groupby("unique_id").cumcount() + 1

    train_mask = sampled_df["row_number"] <= sampled_df["series_length"] - meta.horizon
    train_df = sampled_df.loc[train_mask, ["unique_id", "ds", "y"]].reset_index(drop=True)
    test_df = sampled_df.loc[~train_mask, ["unique_id", "ds", "y"]].reset_index(drop=True)

    static_sample = (
        static_df[static_df["unique_id"].isin(sampled_ids)]
        .sort_values("unique_id")
        .reset_index(drop=True)
    )

    return DatasetBundle(
        train_df=train_df,
        test_df=test_df,
        static_df=static_sample,
        horizon=meta.horizon,
        seasonality=meta.seasonality,
        frequency=meta.freq,
    )
