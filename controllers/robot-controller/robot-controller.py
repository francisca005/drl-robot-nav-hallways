from controller import Supervisor
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
import logging


class CorridorNavigationEnv(Supervisor, gym.Env):
    def __init__(self):
        super().__init__()
        self.timestep = int(self.getBasicTimeStep())

        """
        Define action and observation space.
        """
        self.MAX_SPEED = 5
        self.actions = [
            (self.MAX_SPEED, self.MAX_SPEED),
            (self.MAX_SPEED, self.MAX_SPEED / 2),
            (self.MAX_SPEED / 2, self.MAX_SPEED),
        ]
        self.num_to_action = lambda i: self.actions[i]

        self.action_space = gym.spaces.Discrete(len(self.actions))

        self.n_samples = 360
        self.observation_space = gym.spaces.Box(
            low=0, high=10, shape=(self.n_samples,), dtype=np.float32
        )

        self.end_reward = 30

        self.done = False
        self.truncated = False

        """
        Setup wheels and lidar devices.
        """
        self.wheels = []
        for name in ["left wheel motor", "right wheel motor"]:
            wheel = self.getDevice(name)
            wheel.setPosition(float("inf"))
            wheel.setVelocity(0)
            self.wheels.append(wheel)

        self.lidar = self.getDevice("Lidar")
        self.lidar.enable(self.timestep)
        self.lidar.enablePointCloud()

        self.robot = self.getSelf()
        self.bumper = self.getDevice("Bumper")
        self.bumper.enable(self.timestep)

        self.init_translation = self.robot.getField("translation").getSFVec3f()
        self.init_rotation = self.robot.getField("rotation").getSFRotation()

        self.goal = [1e9, 2.2, 1e9]

        logging.basicConfig(
            filename="logs/turtle0.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

    def reset(self, seed=None):
        print("Resetting environment")
        self.done = False

        self.robot.getField("translation").setSFVec3f(self.init_translation)
        self.robot.getField("rotation").setSFRotation(self.init_rotation)

        for wheel in self.wheels:
            wheel.setVelocity(0)

        self.simulationResetPhysics()
        super().step(self.timestep)

        return self.get_obs(), {}

    def get_obs(self):
        lidar_values = np.array(self.lidar.getRangeImage(), dtype=np.float32)
        return np.clip(lidar_values, 0, 10)

    def calculate_reward(self, lidar_values, action):
        return 0

    def step(self, action):
        l, r = self.num_to_action(action)
        self.wheels[0].setVelocity(l)
        self.wheels[1].setVelocity(r)

        super().step(self.timestep)

        lidar_values = self.get_obs()
        reward = self.calculate_reward(lidar_values, action)

        pos = self.robot.getField("translation").getSFVec3f()
        diff = pos[1] - self.init_translation[1]

        if np.any(diff >= self.goal[1]):
            print("Goal reached")
            self.done = True
            reward += self.end_reward

        if self.bumper.getValue() == 1:
            print("Bumper hit")
            self.done = True
            reward -= self.end_reward

        # TODO: check truncated (time limit)
        self.truncated = False

        if self.done:
            print("Episode done")

        return lidar_values, reward, self.done, self.truncated, {}


def main():
    env = CorridorNavigationEnv()
    model = PPO("MlpPolicy", env, verbose=0, device="cpu")
    model.learn(total_timesteps=100000)


if __name__ == "__main__":
    main()
