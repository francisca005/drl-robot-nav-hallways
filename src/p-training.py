import time
from subprocess import Popen
from wheelchair_env import WheelchairEnv, get_port
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import PPO

if __name__ == "__main__":
    num_robots = 1
    world_file = "../worlds/smart-wheelchairs.wbt"

    for i in range(num_robots):
        port = get_port(i)
        Popen(
            [
                "webots",
                "--batch",
                "--mode=fast",
                f"--port={port}",
                world_file,
            ]
        )

    time.sleep(5)
    print("All processes started")

    env = make_vec_env(
        lambda: WheelchairEnv(id=i),  # Create a unique environment for each robot
        n_envs=num_robots,
        vec_env_cls=SubprocVecEnv,
    )

    model = PPO("MlpPolicy", env, verbose=1, device="cpu")
    model.learn(total_timesteps=100000)
