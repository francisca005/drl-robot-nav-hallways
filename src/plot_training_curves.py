import argparse
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


METRIC_CONFIG = {
    "success_rate": {
        "title": "Training Success Rate",
        "ylabel": "Success rate",
        "output": "success_rate.png",
    },
    "ep_rew_mean": {
        "title": "Mean Episode Reward",
        "ylabel": "Mean episode reward",
        "output": "ep_rew_mean.png",
    },
    "ep_len_mean": {
        "title": "Mean Episode Length",
        "ylabel": "Mean episode length",
        "output": "ep_len_mean.png",
    },
}


def find_csv_for_metric(input_dir: Path, metric: str) -> Path | None:
    """
    Finds a CSV file whose name contains the metric name.
    Example:
        ppo_features.log_ppo-features-run_3_success_rate.csv
        ppo_features.log_ppo-features-run_3-ep_len_mean.csv
    """
    candidates = list(input_dir.glob(f"*{metric}*.csv"))

    if not candidates:
        return None

    # If there are several, use the most recently modified one.
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def read_tensorboard_csv(path: Path) -> pd.DataFrame:
    """
    TensorBoard CSVs usually have columns:
        Wall time, Step, Value
    """
    df = pd.read_csv(path)

    required_columns = {"Step", "Value"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"{path} does not look like a TensorBoard CSV. "
            f"Expected columns include {required_columns}, got {set(df.columns)}"
        )

    df = df[["Step", "Value"]].dropna()
    return df


def plot_metric(df: pd.DataFrame, metric: str, output_dir: Path, experiment_name: str):
    config = METRIC_CONFIG[metric]

    plt.figure(figsize=(8, 5))
    plt.plot(df["Step"], df["Value"], linewidth=2)

    plt.title(f"{experiment_name} - {config['title']}")
    plt.xlabel("Timesteps")
    plt.ylabel(config["ylabel"])
    plt.grid(True)

    output_path = output_dir / config["output"]
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot training curves exported from TensorBoard CSV files."
    )

    parser.add_argument(
        "--experiment",
        choices=["e1_cnn", "e2_features"],
        required=True,
        help="Experiment folder inside results.",
    )

    parser.add_argument(
        "--input_dir",
        default=None,
        help=(
            "Folder containing TensorBoard CSV exports. "
            "If omitted, uses results/<experiment>/tensorboard."
        ),
    )

    args = parser.parse_args()

    project_root = Path.cwd()

    if args.input_dir is None:
        input_dir = project_root / "results" / args.experiment / "tensorboard"
    else:
        input_dir = Path(args.input_dir)

    output_dir = project_root / "results" / args.experiment / "tensorboard"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")

    print(f"Reading TensorBoard CSVs from: {input_dir}")
    print(f"Saving plots to: {output_dir}")

    experiment_name = "E1 CNN" if args.experiment == "e1_cnn" else "E2 Features"

    for metric in METRIC_CONFIG:
        csv_path = find_csv_for_metric(input_dir, metric)

        if csv_path is None:
            print(f"Skipping {metric}: no CSV found.")
            continue

        print(f"Plotting {metric} from {csv_path.name}")
        df = read_tensorboard_csv(csv_path)
        plot_metric(df, metric, output_dir, experiment_name)


if __name__ == "__main__":
    main()