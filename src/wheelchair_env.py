from typing import Tuple, List
import gymnasium as gym
import numpy as np
from sklearn.cluster import DBSCAN
import zmq


class WheelchairEnv(gym.Env):
    def __init__(self, env_id: int):
        super(WheelchairEnv, self).__init__()

        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.bind("ipc:///tmp/giorgio_" + str(env_id))

        self.env_id = env_id

        """ 
        Action and state space definition.
        Robot will be able to control speed of left and right wheels between 0 and 5.
        State is a vector of 360 lidar readings.
        """
        self.obs_shape = (360,)
        self.action_space = gym.spaces.Discrete(5)
        self.to_action = lambda x: (
            [
                [2, 0],  # Forward
                [2, 2],  # Forward and left
                [2, -2],  # Forward and right
                [0.5, 2],  # Left
                [0.5, -2],  # Right
            ]
        )[x]

        self.observation_space = gym.spaces.Box(
            low=0.0, high=10.0, shape=self.obs_shape, dtype=np.float32
        )

        self.prev_obs = self.no_obs()
        self.time_step = 0
        self.time_limit = 6000

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        action = self.to_action(action)
        reward = self.get_reward(self.prev_obs, action)
        done = False
        truncated = False

        obs_done = self.send_action_get_obs(action)
        obs, term = obs_done[:-1], obs_done[-1]

        if term == 1:
            reward += self.collision_reward()
        elif term == 2:
            done = True
            reward += self.goal_reward()

        self.prev_obs = obs
        self.time_step += 1

        if self.time_step >= self.time_limit:
            done = True
            truncated = True
            reward += self.collision_reward()

        return obs, reward, done, truncated, {}

    def get_reward(self, obs: np.ndarray, action: Tuple[int, int]) -> float:
        v = action[0]

        # Reward for moving forward
        r_distance = 1 if v > 0 else -1

        # Collision penalty (if min_range is below a threshold)
        min_range = np.min(obs)
        min_threshold = 0.3
        r_collision = -np.exp(10 * min_range) if min_range < min_threshold else 0

        r_direction = self.direction_reward(obs, action)

        reward = r_distance + r_collision + r_direction
        reward += self.dual_reward(obs)

        return reward

    def direction_reward(self, obs: np.ndarray, action: Tuple[int, int]) -> int:
        v, w = action

        left = np.array(obs[90:178])
        front = np.array(obs[178:183])
        right = np.array(obs[183:270])

        left_max = np.max(left)
        front_max = np.max(front)
        right_max = np.max(right)

        """ Check which direction has the most free space and reward movement in that direction """
        if max(left_max, front_max, right_max) == front_max:
            return 2 if v > 0 else -2
        elif max(left_max, front_max, right_max) == left_max:
            return 2 if w > 0 else -2

        return 2 if w < 0 else -2

    def collision_reward(self) -> int:
        return -100

    def goal_reward(self) -> int:
        """
        Maybe the agent shouldn't receive a reward for reaching the end of the corridor,
        because it will be telling the agent that the observation just before the goal is good,
        when in fact it is not
        """
        return 0

    def dual_reward(self, obs: np.ndarray) -> int:
        """
        Calculates a reward based on the presence of small clusters of points
        directly to the left (80° to 100°) or right (260° to 280°), indicating
        adjacency to another robot.

        :param obs: LIDAR readings (NumPy array of shape (360,))
        :param action: Current action taken by the robot
        :return: Reward value (positive for adjacency)
        """
        clusters = self.cluster_lidar_readings(obs)

        left_range = (80, 100)
        right_range = (260, 280)

        adjacency_reward = -1

        """ Parameters for filtering clusters """
        max_cluster_size = 10
        max_angular_spread = 20
        max_distance_spread = 0.5
        min_length = 0.05
        max_length = 0.15

        for cluster in clusters:
            """Extract distances and angles for the cluster"""
            cluster_distances = obs[cluster]
            cluster_angles = np.radians(cluster)

            """ Convert to Cartesian coordinates """
            x = cluster_distances * np.cos(cluster_angles)
            y = cluster_distances * np.sin(cluster_angles)

            """ Calculate the length of the object that the cluster represents """
            if len(x) > 1:
                length = np.sqrt((x.max() - x.min()) ** 2 + (y.max() - y.min()) ** 2)
            else:
                length = 0

            """ Calculate the mean angle, angular spread, and distance spread of the cluster """
            mean_angle = np.mean(cluster)
            angular_spread = max(cluster) - min(cluster)
            distance_spread = max(cluster_distances) - min(cluster_distances)
            max_dist = max(cluster_distances)
            max_dist_limit = 1

            if (
                len(cluster) <= max_cluster_size
                and angular_spread <= max_angular_spread
                and distance_spread <= max_distance_spread
                and min_length <= length <= max_length
                and max_dist <= max_dist_limit
            ):
                if left_range[0] <= mean_angle <= left_range[1]:
                    print(
                        f"Bot {self.env_id} detected another bot on its leftt with length {length:.2f}m"
                    )
                    print(f"cluster: {cluster}")
                    adjacency_reward = 3
                elif right_range[0] <= mean_angle <= right_range[1]:
                    print(
                        f"Bot {self.env_id} detected another bot on its right with length {length:.2f}m"
                    )
                    print(f"cluster: {cluster}")
                    adjacency_reward = 3

        return adjacency_reward

    def cluster_lidar_readings(
        self, lidar: np.ndarray, fov: int = 360
    ) -> List[List[int]]:
        """
        Clusters LIDAR readings based on spatial proximity.

        :param lidar_readings: NumPy array of shape (360,) with distance values.
        :param fov: Field of view (degrees), typically 360 for each robot.
        :return: List of clusters (each cluster is a list of indices in lidar_readings).
        """
        n_rays = len(lidar)
        angles = np.linspace(-fov / 2, fov / 2, n_rays)

        x = lidar * np.cos(np.radians(angles))
        y = lidar * np.sin(np.radians(angles))
        points = np.column_stack((x, y))

        dbscan = DBSCAN(eps=0.1, min_samples=3)
        labels = dbscan.fit_predict(points)

        clusters = {}
        for i, label in enumerate(labels):
            if label == -1:
                continue  # Ignore noise points
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(i)

        return list(clusters.values())

    def send_action_get_obs(self, action: Tuple[int, int]) -> np.ndarray:
        self.socket.send_pyobj(action)
        return self.get_observation()

    def get_observation(self) -> np.ndarray:
        return np.array(self.socket.recv_pyobj(), dtype=np.float32)

    def no_obs(self) -> np.ndarray:
        return np.full(self.obs_shape, 10.0)

    def reset(self, seed: int = None) -> Tuple[np.ndarray, dict]:
        self.prev_obs = self.no_obs()
        self.time_step = 0
        return self.prev_obs, {}
