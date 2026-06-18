from typing import Tuple

import numpy as np
from gymnasium.spaces import Box

from wheelchair_env import WheelchairEnv
from feature_engineering import FEATURE_SET_DIMS, extract_lidar_features


class WheelchairFeatureEnv(WheelchairEnv):
    """
    Feature-engineering version of the original WheelchairEnv.

    It keeps:
    - same Webots communication;
    - same actions;
    - same reward function;
    - same termination conditions.

    It changes only:
    - observation representation: 360 LiDAR + previous action -> compact features.

    Args:
        env_id: robot index (0-8), passed to WheelchairEnv.
        feature_set: one of "full" (12), "reduced" (8), "directional" (5).
    """

    def __init__(self, env_id: int, feature_set: str = "full"):
        if feature_set not in FEATURE_SET_DIMS:
            raise ValueError(
                f"Unknown feature_set '{feature_set}'. "
                f"Choose from: {list(FEATURE_SET_DIMS)}"
            )

        self.feature_set = feature_set
        # Must be set before super().__init__() because WheelchairEnv.__init__
        # calls self.no_obs(), which needs feature_dim.
        self.feature_dim = FEATURE_SET_DIMS[feature_set]

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

        features = extract_lidar_features(self.prev_lidar, self.prev_action, self.feature_set)
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

        info = {
            "is_success": obs.goal_reached,
            "collision": obs.collided,
            "timeout": truncated and not terminated,
        }

        features = extract_lidar_features(obs.lidar, self.prev_action, self.feature_set)

        return features, reward, terminated, truncated, info
