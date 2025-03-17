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

        self.prev_obs = self.no_obs()

    def step(self, action):
        self.send_action(action)

        reward = self.get_reward(self.prev_obs, action)
        done = False

        # TODO: Implement truncated
        truncated = False

        obs_done = self.get_observation()
        obs, term = obs_done[:-1], obs_done[-1]

        if term == 1:
            reward += self.collision_reward()
        elif term == 2:
            done = True
            reward += self.goal_reward()

        self.prev_obs = obs

        return obs, reward, done, truncated, {}

    def get_reward(self, obs, action):
        v, w = action
        min_range = np.min(obs)

        # Reward for moving forward
        r_distance = 1 if v > 0 else -1

        # Collision penalty (if min_range is below a threshold)
        min_threshold = 0.3  # Define a minimum safe distance
        r_collision = -np.exp(10 * min_range) if min_range < min_threshold else 0

        reward = r_distance + r_collision

        return reward

    def collision_reward(self):
        return -30

    def goal_reward(self):
        return 30

    def send_action(self, action):
        try:
            with open(self.w_path, "wb") as f:
                pickle.dump(action, f)
                f.flush()
        except Exception as e:
            print(f"Error sending action: {e}")

    def get_observation(self):
        try:
            with open(self.r_path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error getting observation: {e}")
            return self.prev_obs

    def no_obs(self):
        return np.full(self.obs_shape, 10.0)

    def reset(self, seed: int = None):
        self.prev_obs = self.no_obs()
        return self.prev_obs, {}
