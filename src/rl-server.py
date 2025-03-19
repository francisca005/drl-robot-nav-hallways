import os
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv
from wheelchair_env import WheelchairEnv

TRAIN_STEPS = 500_000
N_ROBOTS = 2


def train_model():
    try:
        """Start vectorized environment to train model in parallel"""

        def env_fn(i):
            def _init():
                return WheelchairEnv(i)

            return _init

        env = SubprocVecEnv([env_fn(i) for i in range(N_ROBOTS)])

        if os.path.exists("./models/dqn_wheelchair.zip"):
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
            )
        print("Calling learn")
        model.learn(total_timesteps=TRAIN_STEPS)

    finally:
        model.save("./models/dqn_wheelchair")


if __name__ == "__main__":
    train_model()
