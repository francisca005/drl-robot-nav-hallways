"""
E3: Zero-shot evaluation of the E2 features model on sloped corridors.

Usage (from the project root or src/ directory):
    python src/rl-test-features-slope.py --slope 15
    python src/rl-test-features-slope.py --slope 30
    python src/rl-test-features-slope.py --slope 15 --feature-set reduced

Before running, start Webots with the corresponding world:
    slope 15 -> worlds/smart-wheelchairs-slope-15.wbt
    slope 30 -> worlds/smart-wheelchairs-slope-30.wbt

Results are saved to:
    results/e3_slope_zeroshot/evaluation/slope_{N}_metrics.csv
    results/e3_slope_zeroshot/positions_test/positions/{N}/{corridor_id}/t_*.csv
"""

import os
import csv
import shutil
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from wheelchair_feature_env import WheelchairFeatureEnv
from feature_engineering import FEATURE_SET_DIMS


TIME_STEPS = 60_000
N_ROBOTS = 9

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

DATA_POSITIONS_DIR = os.path.join(PROJECT_ROOT, "data", "positions")
RESULTS_BASE = os.path.join(PROJECT_ROOT, "results", "e3_slope_zeroshot")


def _model_paths(feature_set: str):
    tag = f"features_{feature_set}"
    return {
        "model": os.path.join(MODEL_DIR, f"ppo_wheelchair_{tag}"),
        "vecnorm": os.path.join(MODEL_DIR, f"vecnormalize_{tag}.pkl"),
    }


def make_env(env_id: int, feature_set: str):
    def _init():
        return Monitor(WheelchairFeatureEnv(env_id, feature_set=feature_set))

    return _init


def snapshot_existing_positions() -> dict:
    """Record which trajectory files already exist, so we can detect new ones after eval."""
    existing = {}
    for corridor_id in range(N_ROBOTS):
        corridor_dir = os.path.join(DATA_POSITIONS_DIR, str(corridor_id))
        if os.path.exists(corridor_dir):
            existing[corridor_id] = set(os.listdir(corridor_dir))
        else:
            existing[corridor_id] = set()
    return existing


def copy_new_positions(slope: int, existing_before: dict) -> None:
    """Copy trajectory CSVs written during this eval run to the E3 results directory."""
    positions_base = os.path.join(
        RESULTS_BASE, "positions_test", "positions", str(slope)
    )
    print(f"\nCopying position trajectories to {positions_base}")

    for corridor_id in range(N_ROBOTS):
        corridor_src = os.path.join(DATA_POSITIONS_DIR, str(corridor_id))
        corridor_dst = os.path.join(positions_base, str(corridor_id))
        os.makedirs(corridor_dst, exist_ok=True)

        if not os.path.exists(corridor_src):
            print(f"  Corridor {corridor_id}: no position data found, skipping")
            continue

        new_files = set(os.listdir(corridor_src)) - existing_before[corridor_id]
        for fname in sorted(new_files):
            shutil.copy2(
                os.path.join(corridor_src, fname),
                os.path.join(corridor_dst, fname),
            )

        print(f"  Corridor {corridor_id}: {len(new_files)} trajectory file(s) copied")


def run_evaluation(slope: int, feature_set: str = "full") -> None:
    paths = _model_paths(feature_set)

    assert os.path.exists(paths["model"] + ".zip"), (
        f"Feature model not found at {paths['model']}.zip. "
        f"Train E2-{feature_set} first."
    )
    assert os.path.exists(paths["vecnorm"]), (
        f"VecNormalize not found at {paths['vecnorm']}. "
        f"Train E2-{feature_set} first."
    )

    eval_dir = os.path.join(RESULTS_BASE, "evaluation")
    os.makedirs(eval_dir, exist_ok=True)

    existing_positions = snapshot_existing_positions()

    raw_env = SubprocVecEnv([make_env(i, feature_set) for i in range(N_ROBOTS)])
    env = None

    try:
        print(f"Loading E2-{feature_set} VecNormalize from {paths['vecnorm']}")
        env = VecNormalize.load(paths["vecnorm"], raw_env)
        env.training = False
        env.norm_reward = False

        print(f"Loading E2-{feature_set} model from {paths['model']}")
        model = PPO.load(paths["model"], env=env, device="cpu")

        success_counts = [0] * N_ROBOTS
        collision_counts = [0] * N_ROBOTS
        timeout_counts = [0] * N_ROBOTS
        episode_counts = [0] * N_ROBOTS
        total_episode_lengths = [0] * N_ROBOTS

        obs = env.reset()
        print(f"\nStarting E3 zero-shot evaluation — slope {slope}°, {TIME_STEPS} steps\n")

        for step in range(TIME_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)

            for i, done in enumerate(dones):
                if done:
                    episode_counts[i] += 1

                    if infos[i].get("is_success", False):
                        success_counts[i] += 1
                    if infos[i].get("collision", False):
                        collision_counts[i] += 1
                    if infos[i].get("timeout", False):
                        timeout_counts[i] += 1

                    ep_len = infos[i].get("episode", {}).get("l", 0)
                    total_episode_lengths[i] += ep_len

            if (step + 1) % 5_000 == 0:
                print(f"  Step {step + 1}/{TIME_STEPS}")

    finally:
        if env is not None:
            print("Closing evaluation environment")
            env.close()
        else:
            raw_env.close()

    # --- Print summary ---
    print(f"\nE3 zero-shot results — slope {slope}°:")
    for i in range(N_ROBOTS):
        n = episode_counts[i]
        sr = 100 * success_counts[i] / n if n else 0.0
        cr = 100 * collision_counts[i] / n if n else 0.0
        mel = total_episode_lengths[i] / n if n else 0.0
        print(
            f"  Corridor {i:2d}: {success_counts[i]:3d}/{n} success ({sr:6.1f}%)  "
            f"{collision_counts[i]:3d}/{n} collision ({cr:6.1f}%)  "
            f"mean_len {mel:7.1f}"
        )

    # --- Save CSV ---
    results_path = os.path.join(eval_dir, f"slope_{slope}_metrics.csv")
    with open(results_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "corridor_id",
            "slope_deg",
            "successes",
            "collisions",
            "timeouts",
            "episodes",
            "success_rate",
            "collision_rate",
            "mean_episode_length",
        ])
        for i in range(N_ROBOTS):
            n = episode_counts[i]
            sr = 100 * success_counts[i] / n if n else 0.0
            cr = 100 * collision_counts[i] / n if n else 0.0
            mel = total_episode_lengths[i] / n if n else 0.0
            writer.writerow([
                i,
                slope,
                success_counts[i],
                collision_counts[i],
                timeout_counts[i],
                n,
                f"{sr:.2f}",
                f"{cr:.2f}",
                f"{mel:.1f}",
            ])

    print(f"\nSaved metrics to {results_path}")

    copy_new_positions(slope, existing_positions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="E3: Zero-shot evaluation of E2 features model on sloped corridors."
    )
    parser.add_argument(
        "--slope",
        type=int,
        choices=[15, 30],
        required=True,
        help="Slope angle in degrees (15 or 30)",
    )
    parser.add_argument(
        "--feature-set",
        choices=list(FEATURE_SET_DIMS),
        default="full",
        help="E2 feature set variant to use (default: full)",
    )
    args = parser.parse_args()
    run_evaluation(args.slope, feature_set=args.feature_set)
