from typing import Tuple

import numpy as np
from gymnasium.spaces import Box

from feature_engineering import extract_lidar_features
from wheelchair_env import WheelchairEnv


class WheelchairFeatureEnv(WheelchairEnv):
    """
    Feature-engineering version of the original WheelchairEnv.

    It keeps:
    - same Webots communication;
    - same actions;
    - same reward function;
    - same termination conditions.

    It changes only:
    - observation representation: 360 LiDAR + previous action -> 12 handcrafted features.
    """

    def __init__(self, env_id: int):
        # Important: this must be defined before super().__init__(),
        # because WheelchairEnv.__init__() calls self.no_obs().
        self.feature_dim = 12

        super().__init__(env_id)

        self.observation_space = Box(
            low=np.full(self.feature_dim, -1.0, dtype=np.float32),
            high=np.full(self.feature_dim, 1.0, dtype=np.float32),
            dtype=np.float32,
        )

        self.obs_shape = self.observation_space.shape

    def no_obs(self) -> np.ndarray:
        return np.zeros(self.feature_dim, dtype=np.float32)

    def reset(self, seed: int = None, options=None) -> Tuple[np.ndarray, dict]:
        self.prev_lidar = np.zeros(360, dtype=np.float32)
        self.prev_action = 0
        self.time_step = 0
        self.reset_preference()

        features = extract_lidar_features(self.prev_lidar, self.prev_action)

        return features, {}

    def step(self, action: int):
        self.prev_action = action

        action_values = self.to_action(action)

        # Reward is computed using raw LiDAR, exactly as in the baseline.
        reward = self.get_reward(self.prev_lidar, action_values)

        obs = self.send_action_get_obs(action_values)

        if obs.collided:
            reward += self.collision_reward()
        elif obs.goal_reached:
            reward += self.goal_reward()

        self.prev_lidar = obs.lidar
        self.time_step += 1

        terminated = obs.collided or obs.goal_reached
        truncated = self.time_step >= self.time_limit

        if truncated and not terminated:
            reward -= 10
            self.request_timeout_reset()

        info = {
            "is_success": obs.goal_reached,
            "collision": obs.collided,
            "timeout": truncated and not terminated,
        }

        features = extract_lidar_features(obs.lidar, self.prev_action)

        return features, reward, terminated, truncated, info