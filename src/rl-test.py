import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from wheelchair_env import WheelchairEnv
from stable_baselines3.common.monitor import Monitor

TIME_STEPS = 20_000
N_ROBOTS = 9


def run_model():
    """Start vectorized environment to train model in parallel"""

    def env_fn(i):
        def _init():
            return Monitor(WheelchairEnv(i))

        return _init

    env = SubprocVecEnv([env_fn(i) for i in range(N_ROBOTS)])
    env = VecNormalize(env, norm_obs=True, norm_reward=False)

    path = "./models/ppo-good"
    assert os.path.exists(
        path + ".zip"
    ), "Model path does not exist. Please train the model first."

    model = PPO.load(path, env)

    """
    Test the model
    """
    obs = env.reset()
    for _ in range(TIME_STEPS):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)


if __name__ == "__main__":
    run_model()
