"""
Plot training curves comparing E1 (CNN), E2-full, and E2-reduced.

Reads:
  - E1 success rate from results/e1_cnn/tensorboard/e1_ep_success_rate.csv
  - E2-full and E2-reduced from TensorBoard event files in logs/

Saves plots to results/comparison/

Usage:
    python src/plot_comparison_curves.py
"""

import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = SCRIPT_DIR.parent

E1_CSV = PROJECT_ROOT / "results" / "e1_cnn" / "tensorboard" / "e1_ep_success_rate.csv"

E2_FULL_LOG = (
    PROJECT_ROOT / "logs" / "ppo_features_full.log" / "ppo-features_full-run_1"
)
E2_REDUCED_LOG = (
    PROJECT_ROOT / "logs" / "ppo_features_reduced.log" / "ppo-features_reduced-run_5"
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "comparison"


def read_e1_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)[["Step", "Value"]].dropna()
    df.columns = ["step", "success_rate"]
    return df


def read_tb_scalar(log_dir: Path, tag: str) -> pd.DataFrame:
    ea = EventAccumulator(str(log_dir))
    ea.Reload()
    events = ea.Scalars(tag)
    df = pd.DataFrame({"step": [e.step for e in events], "success_rate": [e.value for e in events]})
    return df


def smooth(series: pd.Series, window: int = 3) -> pd.Series:
    return series.rolling(window=window, min_periods=1, center=True).mean()


def plot_comparison(dfs: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    colors = {"E1 (CNN)": "#2196F3", "E2-full (12f)": "#4CAF50", "E2-reduced (8f)": "#FF5722"}

    fig, ax = plt.subplots(figsize=(10, 6))

    for label, df in dfs.items():
        ax.plot(
            df["step"] / 1_000_000,
            smooth(df["success_rate"]),
            label=label,
            color=colors[label],
            linewidth=2,
        )
        ax.plot(
            df["step"] / 1_000_000,
            df["success_rate"],
            color=colors[label],
            linewidth=0.6,
            alpha=0.35,
        )

    ax.set_xlabel("Timesteps (millions)", fontsize=12)
    ax.set_ylabel("Success rate", fontsize=12)
    ax.set_title("Training Success Rate — E1 vs E2-full vs E2-reduced", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    out_path = output_dir / "success_rate_comparison.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_individual(dfs: dict, output_dir: Path) -> None:
    colors = {"E1 (CNN)": "#2196F3", "E2-full (12f)": "#4CAF50", "E2-reduced (8f)": "#FF5722"}
    filenames = {"E1 (CNN)": "e1_success_rate.png", "E2-full (12f)": "e2_full_success_rate.png", "E2-reduced (8f)": "e2_reduced_success_rate.png"}

    for label, df in dfs.items():
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df["step"] / 1_000_000, smooth(df["success_rate"]), color=colors[label], linewidth=2)
        ax.plot(df["step"] / 1_000_000, df["success_rate"], color=colors[label], linewidth=0.6, alpha=0.35)
        ax.set_xlabel("Timesteps (millions)", fontsize=12)
        ax.set_ylabel("Success rate", fontsize=12)
        ax.set_title(f"Training Success Rate — {label}", fontsize=13)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        out_path = output_dir / filenames[label]
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


def plot_bar_comparison(output_dir: Path) -> None:
    data = {
        "Corridor": list(range(9)),
        "E1 (CNN)": [100, 8.2, 100, 100, 18.0, 100, 100, 100, 100],
        "E2-full (12f)": [100, 100, 100, 100, 0, 100, 100, 0, 0],
        "E2-reduced (8f)": [100, 100, 92.9, 100, 0, 6.2, 100, 100, 0],
    }
    df = pd.DataFrame(data).set_index("Corridor")

    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(9)
    width = 0.26
    colors = ["#2196F3", "#4CAF50", "#FF5722"]

    for i, (col, color) in enumerate(zip(df.columns, colors)):
        offset = (i - 1) * width
        bars = ax.bar([xi + offset for xi in x], df[col], width, label=col, color=color, alpha=0.85)

    ax.set_xlabel("Corridor (robot index)", fontsize=12)
    ax.set_ylabel("Success rate (%)", fontsize=12)
    ax.set_title("Evaluation Success Rate per Corridor — E1 vs E2-full vs E2-reduced", fontsize=13)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"Robot {i}" for i in x], rotation=30, ha="right")
    ax.set_ylim(0, 115)
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    out_path = output_dir / "eval_success_rate_per_corridor.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    print("Reading E1 training curve...")
    e1_df = read_e1_csv(E1_CSV)

    print("Reading E2-full training curve...")
    e2_full_df = read_tb_scalar(E2_FULL_LOG, "rollout/success_rate")

    print("Reading E2-reduced training curve...")
    e2_reduced_df = read_tb_scalar(E2_REDUCED_LOG, "rollout/success_rate")

    dfs = {
        "E1 (CNN)": e1_df,
        "E2-full (12f)": e2_full_df,
        "E2-reduced (8f)": e2_reduced_df,
    }

    print(f"\nData points — E1: {len(e1_df)}, E2-full: {len(e2_full_df)}, E2-reduced: {len(e2_reduced_df)}")

    print("\nGenerating plots...")
    plot_comparison(dfs, OUTPUT_DIR)
    plot_individual(dfs, OUTPUT_DIR)
    plot_bar_comparison(OUTPUT_DIR)

    print(f"\nAll plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
