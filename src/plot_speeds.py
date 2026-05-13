import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def sorted_trajectory_files(robot_dir: Path):
    files = list(robot_dir.glob("t_*.csv"))

    def extract_index(path: Path):
        try:
            return int(path.stem.split("_")[1])
        except (IndexError, ValueError):
            return 10**9

    return sorted(files, key=extract_index)


def choose_episode(robot_dir: Path, outcome: str | None):
    files = sorted_trajectory_files(robot_dir)

    for f in files:
        df = pd.read_csv(f)

        if len(df) < 20:
            continue

        if "speed_xy" not in df.columns:
            continue

        if outcome is not None and "outcome" in df.columns:
            file_outcome = str(df["outcome"].iloc[-1])
            if file_outcome != outcome:
                continue

        return f, df

    return None, None


def plot_speed_lines(experiment: str, input_dir: Path, speed_column: str, outcome: str | None):
    output_dir = Path("results") / experiment / "speed_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))

    plotted = 0

    for robot_dir in sorted([p for p in input_dir.iterdir() if p.is_dir() and p.name.isdigit()], key=lambda p: int(p.name)):
        robot_id = int(robot_dir.name)

        trajectory_file, df = choose_episode(robot_dir, outcome)

        if df is None:
            print(f"Robot {robot_id}: no valid episode found")
            continue

        x = df["step"] if "step" in df.columns else range(len(df))
        y = df[speed_column]

        plt.plot(x, y, linewidth=1.4, label=f"Robot {robot_id} ({trajectory_file.stem})")
        plotted += 1

    plt.title(f"{experiment} - {speed_column} over time ({plotted} robots)")
    plt.xlabel("Step")
    plt.ylabel(f"{speed_column} (m/s)")
    plt.grid(True)
    plt.legend(fontsize=8)

    output_path = output_dir / f"{speed_column}_one_episode_per_robot.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved speed plot to: {output_path}")


def plot_mean_speed_bars(experiment: str, input_dir: Path, speed_column: str):
    output_dir = Path("results") / experiment / "speed_plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    robot_ids = []
    mean_speeds = []

    for robot_dir in sorted([p for p in input_dir.iterdir() if p.is_dir() and p.name.isdigit()], key=lambda p: int(p.name)):
        speeds = []

        for f in sorted_trajectory_files(robot_dir):
            df = pd.read_csv(f)

            if len(df) < 20:
                continue

            if speed_column not in df.columns:
                continue

            speeds.extend(df[speed_column].tolist())

        if speeds:
            robot_ids.append(int(robot_dir.name))
            mean_speeds.append(sum(speeds) / len(speeds))

    plt.figure(figsize=(9, 5))
    plt.bar(robot_ids, mean_speeds)

    plt.title(f"{experiment} - mean {speed_column} per robot")
    plt.xlabel("Robot")
    plt.ylabel(f"Mean {speed_column} (m/s)")
    plt.xticks(robot_ids)
    plt.grid(axis="y")

    output_path = output_dir / f"mean_{speed_column}_per_robot.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved mean speed plot to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot robot speed profiles from trajectory CSV files.")

    parser.add_argument("--experiment", required=True)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--speed_column", default="speed_xy", choices=["speed_xy", "speed_3d"])
    parser.add_argument("--outcome", default=None, choices=["success", "collision", "timeout", None])
    parser.add_argument("--bars", action="store_true")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    if args.bars:
        plot_mean_speed_bars(args.experiment, input_dir, args.speed_column)
    else:
        plot_speed_lines(args.experiment, input_dir, args.speed_column, args.outcome)


if __name__ == "__main__":
    main()