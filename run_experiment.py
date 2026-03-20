from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import ExperimentConfig
from src.experiment import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Запуск эксперимента по прогнозированию для HW3.")
    parser.add_argument("--group", default="Monthly", help="Группа M4, по умолчанию: Monthly")
    parser.add_argument("--n-series", type=int, default=120, help="Число рядов в выборке")
    parser.add_argument("--max-clusters", type=int, default=6, help="Максимальное число проверяемых кластеров")
    parser.add_argument("--seed", type=int, default=42, help="Случайное зерно")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Папка для сохраняемых артефактов",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExperimentConfig(
        group=args.group,
        n_series=args.n_series,
        max_clusters=args.max_clusters,
        random_seed=args.seed,
        output_dir=args.output_dir,
    )
    summary = run_experiment(config)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
