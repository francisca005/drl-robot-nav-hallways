from subprocess import Popen
from wheelchair_env import WheelchairEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import PPO

if __name__ == "__main__":
    num_robots = 1
    world_file = "../worlds/smart-wheelchairs.wbt"

    Popen(["webots", "--batch", "--mode=fast", world_file])

    env = make_vec_env(
        lambda i: WheelchairEnv(id=i),  # Create a unique environment for each robot
        n_envs=num_robots,
        vec_env_cls=SubprocVecEnv,
    )

    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=100000)
