import os
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from wheelchair_feature_env import WheelchairFeatureEnv
from feature_engineering import FEATURE_SET_DIMS


TRAIN_STEPS = 3_000_000
N_ROBOTS = 9

MODEL_DIR = "./models"
CHECKPOINT_EVERY_TIMESTEPS = 100_000


def _paths(feature_set: str):
    tag = f"features_{feature_set}"
    return {
        "model": os.path.join(MODEL_DIR, f"ppo_wheelchair_{tag}"),
        "vecnorm": os.path.join(MODEL_DIR, f"vecnormalize_{tag}.pkl"),
        "checkpoint_dir": os.path.join(MODEL_DIR, f"checkpoints_{tag}"),
        "log_dir": f"./logs/ppo_{tag}.log",
    }


class SaveModelAndVecNormalizeCallback(BaseCallback):
    def __init__(self, save_freq_timesteps: int, save_dir: str, tag: str, verbose: int = 1):
        super().__init__(verbose)
        self.save_freq_timesteps = save_freq_timesteps
        self.save_dir = save_dir
        self.tag = tag
        self.last_save_timestep = 0

    def _on_training_start(self) -> None:
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_save_timestep >= self.save_freq_timesteps:
            self.last_save_timestep = self.num_timesteps

            model_path = os.path.join(
                self.save_dir, f"ppo_wheelchair_{self.tag}_step_{self.num_timesteps}"
            )
            vecnorm_path = os.path.join(
                self.save_dir,
                f"vecnormalize_{self.tag}_step_{self.num_timesteps}.pkl",
            )

            self.model.save(model_path)

            if isinstance(self.training_env, VecNormalize):
                self.training_env.save(vecnorm_path)

            if self.verbose:
                print(
                    f"[Checkpoint] Saved {self.tag} model at {self.num_timesteps} timesteps"
                )

        return True


def make_env(env_id: int, feature_set: str):
    def _init():
        return Monitor(WheelchairFeatureEnv(env_id, feature_set=feature_set))

    return _init


def create_vectorized_env(feature_set: str, new: bool, paths: dict):
    raw_env = SubprocVecEnv([make_env(i, feature_set) for i in range(N_ROBOTS)])

    if not new and os.path.exists(paths["vecnorm"]):
        print(f"Loading previous VecNormalize statistics from {paths['vecnorm']}")
        env = VecNormalize.load(paths["vecnorm"], raw_env)
        env.training = True
        env.norm_reward = False
    else:
        print("Creating new VecNormalize statistics")
        env = VecNormalize(raw_env, norm_obs=True, norm_reward=False)

    return env


def create_new_model(env, paths: dict):
    print(f"Creating new feature-engineering model (log: {paths['log_dir']})")

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=8192,
        learning_rate=5e-5,
        batch_size=1024,
        n_epochs=20,
        clip_range=0.1,
        ent_coef=0.01,
        device="cpu",
        tensorboard_log=paths["log_dir"],
    )

    return model


def train_model(feature_set: str = "full", new: bool = False):
    assert feature_set in FEATURE_SET_DIMS, (
        f"Unknown feature_set '{feature_set}'. Choose from: {list(FEATURE_SET_DIMS)}"
    )

    paths = _paths(feature_set)
    tag = f"features_{feature_set}"

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(paths["log_dir"], exist_ok=True)
    os.makedirs(paths["checkpoint_dir"], exist_ok=True)

    env = None
    model = None

    try:
        if new:
            print(f"Starting new E2-{feature_set} training run")

            for path in [paths["model"] + ".zip", paths["vecnorm"]]:
                if os.path.exists(path):
                    print(f"Deleting previous file: {path}")
                    os.remove(path)

        env = create_vectorized_env(feature_set, new=new, paths=paths)

        prev_model_exists = os.path.exists(paths["model"] + ".zip")

        if prev_model_exists and not new:
            print(f"Loading previous model from {paths['model']}")
            model = PPO.load(paths["model"], env=env, device="cpu")
        else:
            model = create_new_model(env, paths)

        checkpoint_callback = SaveModelAndVecNormalizeCallback(
            save_freq_timesteps=CHECKPOINT_EVERY_TIMESTEPS,
            save_dir=paths["checkpoint_dir"],
            tag=tag,
            verbose=1,
        )

        print(f"Starting E2-{feature_set} training ({FEATURE_SET_DIMS[feature_set]} features)")
        model.learn(
            total_timesteps=TRAIN_STEPS,
            tb_log_name=f"ppo-{tag}-run",
            callback=checkpoint_callback,
        )

        print(f"E2-{feature_set} training finished successfully")
        model.save(paths["model"])
        env.save(paths["vecnorm"])

    except KeyboardInterrupt:
        print(f"E2-{feature_set} training interrupted by user")

        if model is not None and env is not None:
            print("Saving interrupted model and VecNormalize statistics")
            model.save(paths["model"] + "_interrupted")
            env.save(
                os.path.join(MODEL_DIR, f"vecnormalize_{tag}_interrupted.pkl")
            )

    finally:
        if env is not None:
            print("Calling env.close()")
            env.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train E2 feature-engineering model")
    parser.add_argument(
        "--feature-set",
        choices=list(FEATURE_SET_DIMS),
        default="full",
        help="Feature set variant to train (default: full)",
    )
    parser.add_argument(
        "--new",
        action="store_true",
        help="Start a new training run, discarding any previous model",
    )
    args = parser.parse_args()

    train_model(feature_set=args.feature_set, new=args.new)
