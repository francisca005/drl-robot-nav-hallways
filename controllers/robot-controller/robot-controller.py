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

        self.robot = self.getFromDef("Turtle0")
        self.bumper = self.getDevice("Bumper")
        self.bumper.enable(self.timestep)

        self.init_pos = self.getSelf().getPosition()
        print("Initial position", self.init_pos)

        self.goal = [1e9, 2.2, 1e9]

        logging.basicConfig(
            filename="logs/turtle0.log",
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
        )

        logging.info("env initialized")

    def reset(self, seed=None):
        self.done = False

        # robot_node = self.getFromDef("Turtle0")
        # if robot_node:
        #     robot_translation = robot_node.getField("translation")
        #     robot_rotation = robot_node.getField("rotation")
        #     robot_translation.setSFVec3f([0.214, 0, 0])
        #     robot_rotation.setSFRotation([0.0, 0.0, 1.0, 0])

        # for wheel in self.__wheels:
        #     wheel.setVelocity(0)

        # Reset robot position
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

        pos = self.getSelf().getPosition()
        diff = pos[1] - self.init_pos[1]

        if np.any(diff >= self.goal[1]):
            print("Goal reached")
            self.done = True

        if self.bumper.getValue() == 1:
            print("Collision detected")
            logging.info("Collision detected")

        lidar_values = self.get_obs()
        reward = self.calculate_reward(lidar_values, action)

        self.done = False
        self.truncated = False

        return lidar_values, reward, self.done, self.truncated, {}


def main():
    env = CorridorNavigationEnv()
    model = PPO("MlpPolicy", env, verbose=0, device="cpu")

    obs, _ = env.reset()
    rewards = []
    acc = 0
    for _ in range(30000):
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        acc += reward
        if done:
            rewards.append(acc)
            acc = 0
            obs, _ = env.reset()


if __name__ == "__main__":
    main()
