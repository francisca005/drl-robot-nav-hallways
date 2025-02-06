from controller import Robot, Motor
import gymnasium
import numpy as np

# import check_env
from gymnasium.utils.env_checker import check_env
import logging


class RobotEnv(gymnasium.Env):
    def __init__(self):
        super(RobotEnv, self).__init__()

        """
        Webots constants
        """
        self.TIME_STEP = 64
        self.MAX_SPEED = 6.28

        """
        Setup robot and devices
        """
        self.robot = Robot()
        self.name = self.robot.getName()
        self.left_motor = self.robot.getDevice("left wheel motor")
        self.right_motor = self.robot.getDevice("right wheel motor")
        self.left_motor.setPosition(float("inf"))
        self.right_motor.setPosition(float("inf"))

        self.lidar = self.robot.getDevice("LDS-01")
        self.lidar.n_samples = 360
        self.lidar.low = np.full(self.lidar.n_samples, 0.0, dtype=np.float32)
        self.lidar.high = np.full(self.lidar.n_samples, 10.0, dtype=np.float32)
        self.lidar.enable(self.TIME_STEP)
        self.lidar.enablePointCloud()

        """
        Setup variables
        """
        self.collided = False
        self.finished = False
        self.timestep = 0
        # self.timelimit = 6000

        self.speed = self.MAX_SPEED / 2

        self.actions = [
            (self.speed, self.speed),  # move forward
            (self.speed, self.speed / 2),  # turn right
            (self.speed / 2, self.speed),  # turn left
        ]

        self.action_space = gymnasium.spaces.Discrete(len(self.actions))
        self.num_to_action = lambda x: self.actions[x]

        self.observation_space = gymnasium.spaces.Box(
            low=self.lidar.low,
            high=self.lidar.high,
            shape=(self.lidar.n_samples,),
            dtype=np.float32,
        )

    def step(self, action):
        pass

    def reset(self):
        pass

    def render(self):
        pass

    def close(self):
        pass

    def seed(self):
        pass

    def get_state(self):
        pass

    def get_reward(self):
        pass

    def get_done(self):
        pass

    def get_info(self):
        pass


def main():
    env = RobotEnv()
    check_env(env)

    logging.basicConfig(
        filename=f"{env.name}.log",
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    while env.robot.step(env.TIME_STEP) != -1:
        pass


if __name__ == "__main__":
    main()
