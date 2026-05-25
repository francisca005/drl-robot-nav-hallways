"""
Plot trajectory comparison figures for key corridors.

Generates one figure per corridor of interest, with subplots showing
the same robot under different models (E1, E2-full, E2-reduced).

Output saved to results/comparison/trajectories/

Usage:
    python src/plot_trajectory_comparison.py
"""

import os
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = SCRIPT_DIR.parent

POSITIONS = {
    "E1 (CNN)":       PROJECT_ROOT / "results" / "e1_cnn"     / "positions_test" / "positions",
    "E2-full (12f)":  PROJECT_ROOT / "results" / "e2_full"    / "positions_test",
    "E2-reduced (8f)":PROJECT_ROOT / "results" / "e2_reduced" / "positions_test",
}

OUTPUT_DIR = PROJECT_ROOT / "results" / "comparison" / "trajectories"

MAX_EPISODES = 8


def load_trajectories(base_dir: Path, robot_id: int, max_eps: int) -> list[pd.DataFrame]:
    folder = base_dir / str(robot_id)
    if not folder.exists():
        return []
    files = sorted(folder.glob("t_*.csv"), key=lambda p: int(p.stem.split("_")[1]))
    dfs = []
    for f in files[:max_eps]:
        try:
            df = pd.read_csv(f)
            if "x" in df.columns and "y" in df.columns and len(df) >= 2:
                dfs.append(df)
        except Exception:
            pass
    return dfs


def plot_panel(ax, trajectories: list[pd.DataFrame], title: str, n_success: int, n_total: int) -> None:
    colors = cm.tab10(np.linspace(0, 0.8, max(len(trajectories), 1)))

    if not trajectories:
        ax.text(0.5, 0.5, "Sem trajectórias\n(episódios por timeout)",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=9, color="gray", style="italic")
        ax.set_title(title, fontsize=10, pad=6)
        ax.axis("off")
        return

    for i, df in enumerate(trajectories):
        ax.plot(df["x"], df["y"], color=colors[i], linewidth=1.2, alpha=0.85)
        ax.scatter(df["x"].iloc[0],  df["y"].iloc[0],  color=colors[i], marker="o", s=20, zorder=3)
        ax.scatter(df["x"].iloc[-1], df["y"].iloc[-1], color=colors[i], marker="x", s=30, zorder=3, linewidths=1.5)

    ax.set_title(title, fontsize=10, pad=6)
    ax.set_xlabel("x (m)", fontsize=8)
    ax.set_ylabel("y (m)", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.25)

    shown = len(trajectories)
    ax.text(0.02, 0.98, f"{n_success}/{n_total} sucesso\n({shown} episódios mostrados)",
            transform=ax.transAxes, fontsize=7.5, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))


def make_comparison_figure(
    robot_id: int,
    models: list[str],
    success_rates: dict,
    total_episodes: dict,
    suptitle: str,
    filename: str,
) -> None:
    n_panels = len(models)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        base = POSITIONS[model]
        trajs = load_trajectories(base, robot_id, MAX_EPISODES)
        sr = success_rates.get(model, 0)
        total = total_episodes.get(model, 0)
        n_succ = round(sr * total / 100)
        title = f"{model}\n{sr:.0f}% sucesso"
        plot_panel(ax, trajs, title, n_succ, total)

    fig.suptitle(suptitle, fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()

    out = OUTPUT_DIR / filename
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Robot 1: E1 falha, E2 melhora drasticamente ---
    make_comparison_figure(
        robot_id=1,
        models=["E1 (CNN)", "E2-full (12f)", "E2-reduced (8f)"],
        success_rates={"E1 (CNN)": 8.2, "E2-full (12f)": 100.0, "E2-reduced (8f)": 100.0},
        total_episodes={"E1 (CNN)": 49, "E2-full (12f)": 30, "E2-reduced (8f)": 30},
        suptitle="Corredor 1 — E1 falha, E2 resolve (● início  × fim)",
        filename="robot1_comparison.png",
    )

    # --- Robot 5: E2-reduced colapsa ---
    make_comparison_figure(
        robot_id=5,
        models=["E1 (CNN)", "E2-full (12f)", "E2-reduced (8f)"],
        success_rates={"E1 (CNN)": 100.0, "E2-full (12f)": 100.0, "E2-reduced (8f)": 18.8},
        total_episodes={"E1 (CNN)": 25, "E2-full (12f)": 17, "E2-reduced (8f)": 16},
        suptitle="Corredor 5 — E2-reduced colapsa (● início  × fim)",
        filename="robot5_comparison.png",
    )

    # --- Robot 7: E2-full timeout, E2-reduced resolve ---
    make_comparison_figure(
        robot_id=7,
        models=["E1 (CNN)", "E2-full (12f)", "E2-reduced (8f)"],
        success_rates={"E1 (CNN)": 100.0, "E2-full (12f)": 0.0, "E2-reduced (8f)": 100.0},
        total_episodes={"E1 (CNN)": 30, "E2-full (12f)": 12, "E2-reduced (8f)": 15},
        suptitle="Corredor 7 — E2-full fica preso, E2-reduced navega (● início  × fim)",
        filename="robot7_comparison.png",
    )

    print(f"\nAll trajectory plots saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
