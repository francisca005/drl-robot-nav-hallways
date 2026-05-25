import os
import csv

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from wheelchair_env import WheelchairEnv


# =========================
# Evaluation configuration
# =========================

TIME_STEPS = 60_000
N_ROBOTS = 9

MODEL_DIR = "./models"
MODEL_PATH = os.path.join(MODEL_DIR, "ppo_wheelchair")
VECNORM_PATH = os.path.join(MODEL_DIR, "vecnormalize.pkl")
RESULTS_PATH = "success_rates.csv"


def make_env(env_id: int):
    def _init():
        return Monitor(WheelchairEnv(env_id))

    return _init


def run_model():
    assert os.path.exists(
        MODEL_PATH + ".zip"
    ), f"Model does not exist at {MODEL_PATH}.zip. Please train the model first."

    assert os.path.exists(
        VECNORM_PATH
    ), f"VecNormalize statistics do not exist at {VECNORM_PATH}. Please train the model first."

    raw_env = SubprocVecEnv([make_env(i) for i in range(N_ROBOTS)])

    env = None

    try:
        print("Loading VecNormalize statistics")
        env = VecNormalize.load(VECNORM_PATH, raw_env)

        # Very important during evaluation:
        # do not update normalization statistics.
        env.training = False
        env.norm_reward = False

        print("Loading trained PPO model")
        model = PPO.load(MODEL_PATH, env=env, device="cpu")

        success_counts = [0] * N_ROBOTS
        episode_counts = [0] * N_ROBOTS

        obs = env.reset()

        print("Starting evaluation")

        for step in range(TIME_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)

            for i, done in enumerate(dones):
                if done:
                    episode_counts[i] += 1

                    if infos[i].get("is_success", False):
                        success_counts[i] += 1

            if (step + 1) % 5_000 == 0:
                print(f"Evaluation step {step + 1}/{TIME_STEPS}")

        print("\nEvaluation results:")

        for i in range(N_ROBOTS):
            if episode_counts[i] > 0:
                rate = 100 * success_counts[i] / episode_counts[i]
                print(
                    f"Robot {i}: {success_counts[i]}/{episode_counts[i]} successes ({rate:.1f}%)"
                )
            else:
                print(f"Robot {i}: no completed episodes.")

        with open(RESULTS_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["robot_id", "successes", "episodes", "success_rate"])

            for i in range(N_ROBOTS):
                rate = (
                    100 * success_counts[i] / episode_counts[i]
                    if episode_counts[i]
                    else 0
                )
                writer.writerow([i, success_counts[i], episode_counts[i], f"{rate:.2f}"])

        print(f"\nSaved results to {RESULTS_PATH}")

    finally:
        if env is not None:
            print("Closing evaluation environment")
            env.close()
        else:
            raw_env.close()


if __name__ == "__main__":
    run_model()