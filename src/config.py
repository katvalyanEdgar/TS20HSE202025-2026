from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data_cache"
RESULTS_DIR = PROJECT_ROOT / "results"


@dataclass(slots=True)
class ExperimentConfig:
    group: str = "Monthly"
    n_series: int = 120
    max_clusters: int = 6
    random_seed: int = 42
    lags: tuple[int, ...] = (1, 2, 3, 6, 12, 18, 24)
    rolling_windows: tuple[int, ...] = (3, 6, 12)
    catboost_iterations: int = 350
    catboost_depth: int = 6
    catboost_learning_rate: float = 0.05
    use_mean_scaling: bool = True
    data_dir: Path = DATA_DIR
    output_dir: Path = RESULTS_DIR

    @property
    def max_lag(self) -> int:
        return max(self.lags)

    @property
    def plots_dir(self) -> Path:
        return self.output_dir / "plots"

    def min_required_length(self, horizon: int) -> int:
        return self.max_lag + horizon + 1
