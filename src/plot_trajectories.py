import argparse
import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def sorted_trajectory_files(positions_dir: Path):
    files = list(positions_dir.glob("t_*.csv"))

    def extract_index(path: Path):
        try:
            return int(path.stem.split("_")[1])
        except (IndexError, ValueError):
            return 10**9

    return sorted(files, key=extract_index)


def plot_robot_trajectories(
    experiment: str,
    robot_id: int,
    max_episodes: int,
    input_dir: Path | None = None,
):
    """
    Plots saved trajectories for one robot.

    Expected folder structure by default:
        results/<experiment>/positions_test/<robot_id>/t_0.csv
        results/<experiment>/positions_test/<robot_id>/t_1.csv
        ...

    Example:
        python src/plot_trajectories.py --experiment e2_features --robot 4 --max_episodes 20
    """

    project_root = Path.cwd()

    if input_dir is None:
        positions_dir = project_root / "results" / experiment / "positions_test" / str(robot_id)
    else:
        positions_dir = Path(input_dir) / str(robot_id)

    if not positions_dir.exists():
        raise FileNotFoundError(f"No trajectory folder found: {positions_dir}")

    files = sorted_trajectory_files(positions_dir)

    if not files:
        raise FileNotFoundError(f"No trajectory CSV files found in: {positions_dir}")

    files = files[:max_episodes]

    output_dir = project_root / "results" / experiment / "trajectories"
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 7))

    for trajectory_file in files:
        df = pd.read_csv(trajectory_file)

        if "x" not in df.columns or "y" not in df.columns:
            print(f"Skipping {trajectory_file.name}: missing x/y columns")
            continue

        if len(df) < 2:
            print(f"Skipping {trajectory_file.name}: too few points")
            continue

        plt.plot(
            df["x"],
            df["y"],
            linewidth=1.5,
            alpha=0.8,
            label=trajectory_file.stem,
        )

        # Start point
        plt.scatter(
            df["x"].iloc[0],
            df["y"].iloc[0],
            marker="o",
            s=25,
        )

        # End point
        plt.scatter(
            df["x"].iloc[-1],
            df["y"].iloc[-1],
            marker="x",
            s=35,
        )

    plt.title(f"{experiment} - Robot {robot_id} trajectories")
    plt.xlabel("x position")
    plt.ylabel("y position")
    plt.axis("equal")
    plt.grid(True)

    if len(files) <= 15:
        plt.legend(fontsize=7)

    output_path = output_dir / f"robot_{robot_id}_trajectories.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved trajectory plot to: {output_path}")


def plot_all_robots(experiment: str, max_episodes: int, input_dir: Path | None = None):
    for robot_id in range(9):
        try:
            plot_robot_trajectories(
                experiment=experiment,
                robot_id=robot_id,
                max_episodes=max_episodes,
                input_dir=input_dir,
            )
        except FileNotFoundError as error:
            print(f"Robot {robot_id}: {error}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot robot trajectories from saved position CSV files."
    )

    parser.add_argument(
        "--experiment",
        choices=["e1_cnn", "e2_features"],
        required=True,
        help="Experiment folder inside results.",
    )

    parser.add_argument(
        "--robot",
        type=int,
        default=None,
        help="Robot id to plot. If omitted, plots all robots 0-8.",
    )

    parser.add_argument(
        "--max_episodes",
        type=int,
        default=20,
        help="Maximum number of trajectories to plot per robot.",
    )

    parser.add_argument(
        "--input_dir",
        default=None,
        help=(
            "Optional custom positions directory. "
            "Expected to contain subfolders 0, 1, ..., 8. "
            "If omitted, uses results/<experiment>/positions_test."
        ),
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir) if args.input_dir else None

    if args.robot is None:
        plot_all_robots(
            experiment=args.experiment,
            max_episodes=args.max_episodes,
            input_dir=input_dir,
        )
    else:
        plot_robot_trajectories(
            experiment=args.experiment,
            robot_id=args.robot,
            max_episodes=args.max_episodes,
            input_dir=input_dir,
        )


if __name__ == "__main__":
    main()