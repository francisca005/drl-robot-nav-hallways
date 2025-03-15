import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from wheelchair_env import WheelchairEnv

TRAIN_STEPS = 100_000
N_ROBOTS = 2


def train_model():
    try:
        for i in range(N_ROBOTS):
            pipe_path = "/tmp/giorgio_" + str(i) + "_"
            os.mkfifo(pipe_path + "obs")
            os.mkfifo(pipe_path + "act")

        """Start vectorized environment to train model in parallel"""

        def env_fn(i):
            def _init():
                return WheelchairEnv(i)

            return _init

        env = SubprocVecEnv([env_fn(i) for i in range(N_ROBOTS)])

        model = PPO("MlpPolicy", env, verbose=1, device="cpu")
        print("Calling learn")
        model.learn(total_timesteps=TRAIN_STEPS)
        model.save("ppo_wheelchair")

    finally:
        for i in range(N_ROBOTS):
            pipe_path = "/tmp/giorgio_" + str(i) + "_"
            os.remove(pipe_path + "obs")
            os.remove(pipe_path + "act")


if __name__ == "__main__":
    train_model()
