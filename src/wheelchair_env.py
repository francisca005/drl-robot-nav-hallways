from multiprocessing import shared_memory
import pickle
import gymnasium as gym
import numpy as np


class WheelchairEnv(gym.Env):
    def __init__(self, env_id: int):
        super(WheelchairEnv, self).__init__()
        self.env_id = env_id
        self.r_path = "/tmp/giorgio_" + str(env_id) + "_act"
        self.w_path = "/tmp/giorgio_" + str(env_id) + "_obs"

        """ 
        Action and state space definition.
        Robot will be able to control speed of left and right wheels between 0 and 5.
        State is a vector of 360 lidar readings.
        """
        self.action_shape = (2,)
        self.obs_shape = (360,)
        self.action_space = gym.spaces.Box(
            low=0.0, high=5.0, shape=self.action_shape, dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=0.0, high=10.0, shape=self.obs_shape, dtype=np.float32
        )

    def step(self, action):
        self.send_action(action)
        print("Sent action")
        obs = self.get_observation()

        reward = 0
        done = False
        truncated = False
        return obs, reward, done, truncated, {}

    def send_action(self, action):
        print("Sending action", action)
        try:
            with open(self.w_path, "wb") as f:
                pickle.dump(action, f)
                f.flush()
        except Exception as e:
            print(f"Error sending action: {e}")

    def get_observation(self):
        print("Waiting for observation")
        try:
            with open(self.r_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error getting observation: {e}")

    def reset(self, seed: int = None):
        return np.zeros(self.obs_shape), {}
