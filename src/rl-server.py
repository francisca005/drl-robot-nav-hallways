import os
import sys
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv
from wheelchair_env import WheelchairEnv

TRAIN_STEPS = 500_000
N_ROBOTS = 4


def train_model(new=False):
    try:
        """Start vectorized environment to train model in parallel"""

        def env_fn(i):
            def _init():
                return WheelchairEnv(i)

            return _init

        env = SubprocVecEnv([env_fn(i) for i in range(N_ROBOTS)])

        prev_model = os.path.exists("./models/dqn_wheelchair.zip")

        if new and prev_model:
            print("Deleting previous model")
            os.remove("./models/dqn_wheelchair.zip")
        elif prev_model:
            print("Loading previous model")
            model = DQN.load("./models/dqn_wheelchair", env)
        else:
            print("Creating new model")
            model = DQN(
                "MlpPolicy",
                env,
                verbose=1,
                gamma=0.9999,
                batch_size=200,
                exploration_fraction=0.25,
            )

        model.learn(total_timesteps=TRAIN_STEPS)

    finally:
        model.save("./models/dqn_wheelchair")


if __name__ == "__main__":
    new = sys.argv[1] == "--new" if len(sys.argv) > 1 else False
    train_model(new)
