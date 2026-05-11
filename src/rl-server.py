import os
import sys

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from wheelchair_env import WheelchairEnv
from cnn_feature_extractor import LidarCNNFeatureExtractor


# =========================
# Training configuration
# =========================

TRAIN_STEPS = 3_000_000
N_ROBOTS = 9

MODEL_DIR = "./models"
LOG_DIR = "./logs/ppo.log"
CHECKPOINT_DIR = "./models/checkpoints"

MODEL_PATH = os.path.join(MODEL_DIR, "ppo_wheelchair")
VECNORM_PATH = os.path.join(MODEL_DIR, "vecnormalize.pkl")

# Saves intermediate model + normalization every X real environment timesteps.
# This protects you from losing a long training run if something crashes.
CHECKPOINT_EVERY_TIMESTEPS = 100_000


class SaveModelAndVecNormalizeCallback(BaseCallback):
    """
    Periodically saves both:
    1. the PPO model;
    2. the VecNormalize statistics.

    This is important because the model depends on the observation normalization
    statistics learned during training.
    """

    def __init__(self, save_freq_timesteps: int, save_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.save_freq_timesteps = save_freq_timesteps
        self.save_dir = save_dir
        self.last_save_timestep = 0

    def _on_training_start(self) -> None:
        os.makedirs(self.save_dir, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_save_timestep >= self.save_freq_timesteps:
            self.last_save_timestep = self.num_timesteps

            model_path = os.path.join(
                self.save_dir, f"ppo_wheelchair_step_{self.num_timesteps}"
            )
            vecnorm_path = os.path.join(
                self.save_dir, f"vecnormalize_step_{self.num_timesteps}.pkl"
            )

            self.model.save(model_path)

            if isinstance(self.training_env, VecNormalize):
                self.training_env.save(vecnorm_path)

            if self.verbose:
                print(f"[Checkpoint] Saved model and VecNormalize at {self.num_timesteps} timesteps")

        return True


def make_env(env_id: int):
    def _init():
        return Monitor(WheelchairEnv(env_id))

    return _init


def create_vectorized_env(new: bool):
    """
    Creates the vectorized environment.

    If continuing a previous training run, it loads the previous VecNormalize
    statistics. If starting from scratch, it creates a new VecNormalize wrapper.
    """

    raw_env = SubprocVecEnv([make_env(i) for i in range(N_ROBOTS)])

    if not new and os.path.exists(VECNORM_PATH):
        print("Loading previous VecNormalize statistics")
        env = VecNormalize.load(VECNORM_PATH, raw_env)
        env.training = True
        env.norm_reward = False
    else:
        print("Creating new VecNormalize statistics")
        env = VecNormalize(raw_env, norm_obs=True, norm_reward=False)

    return env


def create_new_model(env):
    print("Creating new model")

    policy_kwargs = dict(
        features_extractor_class=LidarCNNFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=128),
    )

    model = PPO(
        "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        n_steps=8192,
        learning_rate=5e-5,
        batch_size=1024,
        n_epochs=20,
        clip_range=0.1,
        ent_coef=0.01,
        device="cpu",
        tensorboard_log=LOG_DIR,
    )

    return model


def train_model(new: bool = False):
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    env = None
    model = None

    try:
        if new:
            print("Starting new training run")

            if os.path.exists(MODEL_PATH + ".zip"):
                print("Deleting previous model")
                os.remove(MODEL_PATH + ".zip")

            if os.path.exists(VECNORM_PATH):
                print("Deleting previous VecNormalize statistics")
                os.remove(VECNORM_PATH)

        env = create_vectorized_env(new=new)

        prev_model_exists = os.path.exists(MODEL_PATH + ".zip")

        if prev_model_exists and not new:
            print("Loading previous model")
            model = PPO.load(MODEL_PATH, env=env, device="cpu")
        else:
            model = create_new_model(env)

        checkpoint_callback = SaveModelAndVecNormalizeCallback(
            save_freq_timesteps=CHECKPOINT_EVERY_TIMESTEPS,
            save_dir=CHECKPOINT_DIR,
            verbose=1,
        )

        print("Starting training")
        model.learn(
            total_timesteps=TRAIN_STEPS,
            tb_log_name="ppo-run",
            callback=checkpoint_callback,
        )

        print("Training finished successfully")
        print("Saving final model and VecNormalize statistics")

        model.save(MODEL_PATH)
        env.save(VECNORM_PATH)

    except KeyboardInterrupt:
        print("Training interrupted by user")

        if model is not None and env is not None:
            print("Saving interrupted model and VecNormalize statistics")
            model.save(MODEL_PATH + "_interrupted")
            env.save(os.path.join(MODEL_DIR, "vecnormalize_interrupted.pkl"))

    finally:
        if env is not None:
            print("Calling env.close()")
            env.close()


if __name__ == "__main__":
    new = sys.argv[1] == "--new" if len(sys.argv) > 1 else False
    train_model(new)