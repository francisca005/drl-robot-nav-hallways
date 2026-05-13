import argparse
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
    max_episodes: int | None,
    input_dir: Path | None = None,
):
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

    if max_episodes is not None:
        files = files[:max_episodes]

    output_dir = project_root / "results" / experiment / "trajectories"
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 7))

    plotted = 0

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
            linewidth=1.2,
            alpha=0.55,
            label=trajectory_file.stem,
        )

        plt.scatter(df["x"].iloc[0], df["y"].iloc[0], marker="o", s=20)
        plt.scatter(df["x"].iloc[-1], df["y"].iloc[-1], marker="x", s=30)

        plotted += 1

    plt.title(f"{experiment} - Robot {robot_id} trajectories ({plotted} episodes)")
    plt.xlabel("x position")
    plt.ylabel("y position")
    plt.axis("equal")
    plt.grid(True)

    if plotted <= 15:
        plt.legend(fontsize=7)

    output_path = output_dir / f"robot_{robot_id}_trajectories.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved {plotted} trajectories to: {output_path}")


def discover_robot_ids(input_base_dir: Path):
    robot_ids = []

    if not input_base_dir.exists():
        return robot_ids

    for folder in input_base_dir.iterdir():
        if folder.is_dir() and folder.name.isdigit():
            robot_ids.append(int(folder.name))

    return sorted(robot_ids)


def plot_all_robots(experiment: str, max_episodes: int | None, input_dir: Path | None = None):
    project_root = Path.cwd()

    if input_dir is None:
        input_base_dir = project_root / "results" / experiment / "positions_test"
    else:
        input_base_dir = Path(input_dir)

    robot_ids = discover_robot_ids(input_base_dir)

    if not robot_ids:
        print(f"No robot folders found in: {input_base_dir}")
        return

    for robot_id in robot_ids:
        try:
            plot_robot_trajectories(
                experiment=experiment,
                robot_id=robot_id,
                max_episodes=max_episodes,
                input_dir=input_dir,
            )
        except FileNotFoundError as error:
            print(f"Robot {robot_id}: {error}")


def parse_max_episodes(value: str):
    if value.lower() in ["all", "none", "-1"]:
        return None

    parsed = int(value)

    if parsed <= 0:
        return None

    return parsed


def main():
    parser = argparse.ArgumentParser(
        description="Plot robot trajectories from saved position CSV files."
    )

    parser.add_argument(
        "--experiment",
        required=True,
        help="Experiment folder inside results, e.g. e1_cnn, e2_features, e1_10_slope.",
    )

    parser.add_argument(
        "--robot",
        type=int,
        default=None,
        help="Robot id to plot. If omitted, plots every robot folder found.",
    )

    parser.add_argument(
        "--max_episodes",
        default="all",
        help="Maximum number of trajectories per robot, or 'all'. Default: all.",
    )

    parser.add_argument(
        "--input_dir",
        default=None,
        help=(
            "Optional custom positions directory. "
            "Expected to contain robot subfolders such as 0, 1, ..., 8. "
            "If omitted, uses results/<experiment>/positions_test."
        ),
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir) if args.input_dir else None
    max_episodes = parse_max_episodes(args.max_episodes)

    if args.robot is None:
        plot_all_robots(
            experiment=args.experiment,
            max_episodes=max_episodes,
            input_dir=input_dir,
        )
    else:
        plot_robot_trajectories(
            experiment=args.experiment,
            robot_id=args.robot,
            max_episodes=max_episodes,
            input_dir=input_dir,
        )


if __name__ == "__main__":
    main()