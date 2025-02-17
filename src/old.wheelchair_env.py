from controller import Supervisor
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
import logging


class WheelchairEnv(gym.Env):
    def __init__(self, id: int = 0):
        super().__init__()
        self.sv = Supervisor()
        self.webots_timestep = int(self.sv.getBasicTimeStep())

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

        self.timestep = 0
        self.end_reward = 30
        self.timelimit = 1000

        """
        Setup wheels and lidar devices.
        """
        self.wheels = []
        for name in ["left wheel motor", "right wheel motor"]:
            wheel = self.sv.getDevice(name)
            wheel.setPosition(float("inf"))
            wheel.setVelocity(0)
            self.wheels.append(wheel)

        self.robot = self.sv.getFromDef(f"Giorgio_{id}")

        self.lidar = self.sv.getDevice(f"Lidar_{id}")
        self.lidar.enable(self.webots_timestep)
        self.lidar.enablePointCloud()

        self.bumper = self.sv.getDevice(f"Bumper_{id}")
        self.bumper.enable(self.webots_timestep)

        self.init_translation = self.robot.getField("translation").getSFVec3f()
        self.init_rotation = self.robot.getField("rotation").getSFRotation()

        self.goal = [1e9, 2.2, 1e9]

        logging.basicConfig(
            filename=f"logs/giorgio_{id}.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

    def reset(self, seed=None):
        self.timestep = 0

        self.robot.getField("translation").setSFVec3f(self.init_translation)
        self.robot.getField("rotation").setSFRotation(self.init_rotation)

        for wheel in self.wheels:
            wheel.setVelocity(0)

        self.sv.simulationResetPhysics()
        self.sv.step(self.webots_timestep)

        return self.get_obs(), {}

    def get_obs(self):
        lidar_values = np.array(self.lidar.getRangeImage(), dtype=np.float32)
        return np.clip(lidar_values, 0, 10)

    def calculate_reward(self, lidar, action):
        lidar = np.array(lidar, dtype=np.float32)
        lidar = np.clip(lidar, 0, 10)
        lidar = lidar[70:290]  # consider only front lidar values

        reward = -0.02  # incentive to move quicker
        threshold = 0.5  # threshold for min distance to a wall to give negative reward

        min_dist = np.min(lidar)
        if min_dist <= threshold:
            reward -= np.exp(-min_dist)

        # TODO: consider (previous) action for reward

        return reward

    def step(self, action):
        self.timestep += 1
        done = False
        truncated = False

        l, r = self.num_to_action(action)
        self.wheels[0].setVelocity(l)
        self.wheels[1].setVelocity(r)

        self.sv.step(self.webots_timestep)

        lidar_values = self.get_obs()
        reward = self.calculate_reward(lidar_values, action)

        pos = self.robot.getField("translation").getSFVec3f()
        diff = pos[1] - self.init_translation[1]

        if np.any(diff >= self.goal[1]):
            reward += self.end_reward
            done = True
            logging.info(f"Goal reached in {self.timestep} steps with reward {reward}")
        elif self.bumper.getValue() == 1:
            reward -= self.end_reward
            done = True
            logging.info(
                f"Episode terminated in {self.timestep} steps due to collision with reward {reward}"
            )
        elif self.timestep >= self.timelimit:
            reward -= self.end_reward
            done = True
            truncated = True
            logging.info(
                f"Episode terminated in {self.timestep} steps due to time limit with reward {reward}"
            )

        return lidar_values, reward, done, truncated, {}


def main():
    env = WheelchairEnv()
    model = PPO("MlpPolicy", env, verbose=0, device="cpu")
    model.learn(total_timesteps=200000)


def get_port(i: int) -> int:
    return 1500 + i


if __name__ == "__main__":
    main()
