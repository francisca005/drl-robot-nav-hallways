"""
Plot training curves comparing E1 (CNN), E2-full, E2-reduced, and E2-directional.

Reads:
  - E1 success rate from results/e1_cnn/tensorboard/e1_ep_success_rate.csv
  - E2 variants from TensorBoard event files in logs/

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
    PROJECT_ROOT / "logs" / "ppo_features_full.log" / "ppo-features_full-run_2"
)
E2_REDUCED_LOG = (
    PROJECT_ROOT / "logs" / "ppo_features_reduced.log" / "ppo-features_reduced-run_7"
)
E2_DIRECTIONAL_LOG = (
    PROJECT_ROOT / "logs" / "ppo_features_directional.log" / "ppo-features_directional-run_2"
)

OUTPUT_DIR = PROJECT_ROOT / "results" / "comparison"

COLORS = {
    "E1 (CNN)":              "#2196F3",
    "E2-full (12f)":         "#4CAF50",
    "E2-reduced (8f)":       "#FF5722",
    "E2-directional (5f)":   "#9C27B0",
}

FILENAMES = {
    "E1 (CNN)":            "e1_success_rate.png",
    "E2-full (12f)":       "e2_full_success_rate.png",
    "E2-reduced (8f)":     "e2_reduced_success_rate.png",
    "E2-directional (5f)": "e2_directional_success_rate.png",
}


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

    fig, ax = plt.subplots(figsize=(10, 6))

    for label, df in dfs.items():
        ax.plot(
            df["step"] / 1_000_000,
            smooth(df["success_rate"]),
            label=label,
            color=COLORS[label],
            linewidth=2,
        )
        ax.plot(
            df["step"] / 1_000_000,
            df["success_rate"],
            color=COLORS[label],
            linewidth=0.6,
            alpha=0.35,
        )

    ax.set_xlabel("Timesteps (millions)", fontsize=12)
    ax.set_ylabel("Success rate", fontsize=12)
    ax.set_title("Training Success Rate — E1 vs E2 variants", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    out_path = output_dir / "success_rate_comparison.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_individual(dfs: dict, output_dir: Path) -> None:
    for label, df in dfs.items():
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(df["step"] / 1_000_000, smooth(df["success_rate"]), color=COLORS[label], linewidth=2)
        ax.plot(df["step"] / 1_000_000, df["success_rate"], color=COLORS[label], linewidth=0.6, alpha=0.35)
        ax.set_xlabel("Timesteps (millions)", fontsize=12)
        ax.set_ylabel("Success rate", fontsize=12)
        ax.set_title(f"Training Success Rate — {label}", fontsize=13)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        out_path = output_dir / FILENAMES[label]
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")


def plot_bar_comparison(output_dir: Path) -> None:
    data = {
        "Corridor": list(range(9)),
        "E1 (CNN)":            [100, 8.16, 100, 100, 17.95, 100, 100, 100, 100],
        "E2-full (12f)":       [100, 100,  100, 100,  0,    100, 100, 100, 100],
        "E2-reduced (8f)":     [100, 100,  100, 100,  0,      0, 100,  60, 100],
        "E2-directional (5f)": [100, 100,  100, 100,  0,    100,   0, 100, 100],
    }
    df = pd.DataFrame(data).set_index("Corridor")

    fig, ax = plt.subplots(figsize=(14, 6))
    x = range(9)
    width = 0.2
    labels = list(df.columns)

    for i, col in enumerate(labels):
        offset = (i - 1.5) * width
        ax.bar([xi + offset for xi in x], df[col], width, label=col, color=COLORS[col], alpha=0.85)

    ax.set_xlabel("Corridor (robot index)", fontsize=12)
    ax.set_ylabel("Success rate (%)", fontsize=12)
    ax.set_title("Evaluation Success Rate per Corridor — E1 vs E2 variants", fontsize=13)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"Robot {i}" for i in x], rotation=30, ha="right")
    ax.set_ylim(0, 120)
    ax.legend(fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    out_path = output_dir / "eval_success_rate_per_corridor.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_mean_bar(output_dir: Path) -> None:
    means = {
        "E1 (CNN)":            (100 + 8.16 + 100 + 100 + 17.95 + 100 + 100 + 100 + 100) / 9,
        "E2-full (12f)":       (100 + 100 + 100 + 100 + 0 + 100 + 100 + 100 + 100) / 9,
        "E2-reduced (8f)":     (100 + 100 + 100 + 100 + 0 + 0 + 100 + 60 + 100) / 9,
        "E2-directional (5f)": (100 + 100 + 100 + 100 + 0 + 100 + 0 + 100 + 100) / 9,
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = list(means.keys())
    values = list(means.values())
    colors = [COLORS[l] for l in labels]

    bars = ax.bar(labels, values, color=colors, alpha=0.85, width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("Mean success rate (%)", fontsize=12)
    ax.set_title("Mean Success Rate across 9 Corridors", fontsize=13)
    ax.set_ylim(0, 115)
    ax.grid(True, axis="y", alpha=0.3)
    plt.xticks(rotation=15, ha="right", fontsize=10)

    out_path = output_dir / "mean_success_rate.png"
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

    print("Reading E2-directional training curve...")
    e2_dir_df = read_tb_scalar(E2_DIRECTIONAL_LOG, "rollout/success_rate")

    dfs = {
        "E1 (CNN)":            e1_df,
        "E2-full (12f)":       e2_full_df,
        "E2-reduced (8f)":     e2_reduced_df,
        "E2-directional (5f)": e2_dir_df,
    }

    print(f"\nData points — E1: {len(e1_df)}, E2-full: {len(e2_full_df)}, "
          f"E2-reduced: {len(e2_reduced_df)}, E2-directional: {len(e2_dir_df)}")

    print("\nGenerating plots...")
    plot_comparison(dfs, OUTPUT_DIR)
    plot_individual(dfs, OUTPUT_DIR)
    plot_bar_comparison(OUTPUT_DIR)
    plot_mean_bar(OUTPUT_DIR)

    print(f"\nAll plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
