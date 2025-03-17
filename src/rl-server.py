import os
from stable_baselines3 import PPO
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

        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            device="cpu",
            n_steps=5000,
            gamma=0.9999,
            batch_size=200,
        )
        print("Calling learn")
        model.learn(total_timesteps=TRAIN_STEPS)

    finally:
        model.save("ppo_wheelchair")


if __name__ == "__main__":
    train_model()
