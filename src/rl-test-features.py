import argparse
import csv
import os
import shutil

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


def _paths(feature_set: str):
    tag = f"features_{feature_set}"
    results_base = os.path.join(PROJECT_ROOT, "results", f"e2_{feature_set}")
    return {
        "model": os.path.join(MODEL_DIR, f"ppo_wheelchair_{tag}"),
        "vecnorm": os.path.join(MODEL_DIR, f"vecnormalize_{tag}.pkl"),
        "results": os.path.join(results_base, "evaluation", "metrics.csv"),
        "positions_dst": os.path.join(results_base, "positions_test"),
    }


def make_env(env_id: int, feature_set: str):
    def _init():
        return Monitor(WheelchairFeatureEnv(env_id, feature_set=feature_set))

    return _init


def clear_positions() -> None:
    """Delete all robot subfolders from data/positions/ before evaluation."""
    for robot_id in range(N_ROBOTS):
        corridor_dir = os.path.join(DATA_POSITIONS_DIR, str(robot_id))
        if os.path.exists(corridor_dir):
            shutil.rmtree(corridor_dir)
    print(f"Cleared {DATA_POSITIONS_DIR}")


def copy_positions(positions_dst: str) -> None:
    """Copy data/positions/ to results directory after evaluation."""
    print(f"\nCopying trajectories to {positions_dst}")
    if os.path.exists(positions_dst):
        shutil.rmtree(positions_dst)
    shutil.copytree(DATA_POSITIONS_DIR, positions_dst)
    for robot_id in range(N_ROBOTS):
        dst = os.path.join(positions_dst, str(robot_id))
        count = len(os.listdir(dst)) if os.path.exists(dst) else 0
        print(f"  Robot {robot_id}: {count} trajectory file(s) copied")


def run_model(feature_set: str = "full"):
    assert feature_set in FEATURE_SET_DIMS, (
        f"Unknown feature_set '{feature_set}'. Choose from: {list(FEATURE_SET_DIMS)}"
    )

    paths = _paths(feature_set)

    assert os.path.exists(paths["model"] + ".zip"), (
        f"Model not found at {paths['model']}.zip. Train E2-{feature_set} first."
    )
    assert os.path.exists(paths["vecnorm"]), (
        f"VecNormalize not found at {paths['vecnorm']}. Train E2-{feature_set} first."
    )

    clear_positions()

    raw_env = SubprocVecEnv([make_env(i, feature_set) for i in range(N_ROBOTS)])
    env = None

    try:
        print(f"Loading VecNormalize from {paths['vecnorm']}")
        env = VecNormalize.load(paths["vecnorm"], raw_env)
        env.training = False
        env.norm_reward = False

        print(f"Loading model from {paths['model']}")
        model = PPO.load(paths["model"], env=env, device="cpu")

        success_counts = [0] * N_ROBOTS
        collision_counts = [0] * N_ROBOTS
        timeout_counts = [0] * N_ROBOTS
        episode_counts = [0] * N_ROBOTS
        episode_length_sums = [0] * N_ROBOTS
        # track current episode length per robot
        current_lengths = [0] * N_ROBOTS

        obs = env.reset()

        print(f"Starting E2-{feature_set} evaluation ({TIME_STEPS} steps)")

        for step in range(TIME_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)

            for i in range(N_ROBOTS):
                current_lengths[i] += 1

                if dones[i]:
                    episode_counts[i] += 1
                    episode_length_sums[i] += current_lengths[i]
                    current_lengths[i] = 0

                    if infos[i].get("is_success", False):
                        success_counts[i] += 1
                    elif infos[i].get("collision", False):
                        collision_counts[i] += 1
                    elif infos[i].get("timeout", False):
                        timeout_counts[i] += 1

            if (step + 1) % 5_000 == 0:
                print(f"E2-{feature_set} evaluation step {step + 1}/{TIME_STEPS}")

        print(f"\nE2-{feature_set} evaluation results:")
        print(
            f"{'Robot':>6}  {'Episodes':>8}  {'Success%':>9}  "
            f"{'Collision%':>11}  {'Timeout%':>9}  {'MeanLen':>8}"
        )

        for i in range(N_ROBOTS):
            n = episode_counts[i]
            if n > 0:
                sr = 100.0 * success_counts[i] / n
                cr = 100.0 * collision_counts[i] / n
                tr = 100.0 * timeout_counts[i] / n
                ml = episode_length_sums[i] / n
                print(
                    f"{i:>6}  {n:>8}  {sr:>9.1f}  {cr:>11.1f}  {tr:>9.1f}  {ml:>8.1f}"
                )
            else:
                print(f"{i:>6}  no completed episodes")

        os.makedirs(os.path.dirname(paths["results"]), exist_ok=True)
        with open(paths["results"], "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "robot_id", "episodes", "successes", "collisions", "timeouts",
                "success_rate", "collision_rate", "timeout_rate", "mean_episode_length",
            ])

            for i in range(N_ROBOTS):
                n = episode_counts[i]
                sr = 100.0 * success_counts[i] / n if n else 0.0
                cr = 100.0 * collision_counts[i] / n if n else 0.0
                tr = 100.0 * timeout_counts[i] / n if n else 0.0
                ml = episode_length_sums[i] / n if n else 0.0
                writer.writerow([
                    i, n,
                    success_counts[i], collision_counts[i], timeout_counts[i],
                    f"{sr:.2f}", f"{cr:.2f}", f"{tr:.2f}", f"{ml:.1f}",
                ])

        print(f"\nSaved E2-{feature_set} results to {paths['results']}")
        copy_positions(paths["positions_dst"])

    finally:
        if env is not None:
            print(f"Closing E2-{feature_set} evaluation environment")
            env.close()
        else:
            raw_env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate E2 feature-engineering model")
    parser.add_argument(
        "--feature-set",
        choices=list(FEATURE_SET_DIMS),
        default="full",
        help="Feature set variant to evaluate (default: full)",
    )
    args = parser.parse_args()

    run_model(feature_set=args.feature_set)
