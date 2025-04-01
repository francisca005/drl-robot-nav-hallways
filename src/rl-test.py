import os
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv
from wheelchair_env import WheelchairEnv

TRAIN_STEPS = 10_000_000
N_ROBOTS = 4


def train_model():
    """Start vectorized environment to train model in parallel"""

    def env_fn(i):
        def _init():
            return WheelchairEnv(i)

        return _init

    env = SubprocVecEnv([env_fn(i) for i in range(N_ROBOTS)])

    assert os.path.exists("./models/dqn_wheelchair.zip")

    model = DQN.load("./models/dqn_wheelchair", env)

    """
    Test the model
    """
    obs = env.reset()
    for _ in range(20_000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)


if __name__ == "__main__":
    train_model()
