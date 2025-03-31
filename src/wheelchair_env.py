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
        self.obs_shape = (28,)
        self.action_space = gym.spaces.Discrete(6)

        """
        Lidar readings are divided into 8 regions, each covering 20 degrees in front of the robot.
        Left and right readings are included to check for adjacency to other robots.
        """
        self.regions = [(90 + i * 20, 110 + i * 20) for i in range(8)]
        self.left = (85, 95)
        self.left_index = (8, 18)
        self.right = (265, 275)
        self.right_index = (18, 28)

        v = 4
        w = 1
        self.to_action = lambda x: (
            [
                [v, 0],  # Forward
                [v, w],  # Forward and left
                [v, -w],  # Forward and right
                [0, w],  # Left
                [0, -w],  # Right
                [0, 0],  # Stop
            ]
        )[x]

        self.observation_space = gym.spaces.Box(
            low=0.0, high=10.0, shape=self.obs_shape, dtype=np.float32
        )

        self.prev_obs = self.no_obs()
        self.time_step = 0
        self.time_limit = 6000

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Takes an action and returns the next observation, reward, done flag, and info.
        :param action: Action to be taken (int)
        :return: Tuple of (observation, reward, done, info)
        """

        action = self.to_action(action)
        reward = self.get_reward(self.prev_obs, action)

        done = False
        truncated = False

        obs, term = self.send_action_get_obs(action)

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
            print(f"Time limit reached for robot {self.env_id}")

        return obs, reward, done, truncated, {}

    def get_reward(self, obs: np.ndarray, action: Tuple[int, int]) -> float:
        v = action[0]

        """ Reward for moving forward """
        r_distance = 1 if v > 0 else -1

        """ Collision penalty, exponential on distance if below a threshold """
        min_range = np.min(obs)
        min_threshold = 0.3
        r_collision = -np.exp(3 * (1 - min_range)) if min_range < min_threshold else 0

        r_direction = self.direction_reward(obs, action)

        r_dual = self.dual_reward(obs)
        reward = r_distance + r_collision + r_direction + r_dual

        return reward

    def direction_reward(self, obs: np.ndarray, action: Tuple[int, int]) -> int:
        v, w = action

        half = len(self.regions) // 2
        left = obs[half]
        front = obs[half + 1]
        right = obs[half + 2]

        """ Check which direction has the most free space and reward movement in that direction """
        r = 2
        if max(left, front, right) == front:
            return r if v > 0 else -r
        elif max(left, front, right) == left:
            return r if w > 0 else -r

        return r if w < 0 else -r

    def collision_reward(self) -> int:
        return -10

    def goal_reward(self) -> int:
        """
        Maybe the agent shouldn't receive a reward for reaching the end of the corridor,
        because it will be telling the agent that the observation just before the goal is good,
        when in fact it is not
        """
        return 0

    def dual_reward(self, obs: np.ndarray) -> int:
        """
        Calculates a reward based on the detection of another robot to the left or right of this robot.

        :param obs: LIDAR readings (NumPy array of shape (360,))
        :param action: Current action taken by the robot
        :return: Reward value (positive for adjacency)
        """

        ls, le = self.left_index
        rs, re = self.right_index

        clusters = [obs[ls:le], obs[rs:re]]

        adjacency_reward = 0  # Initialize to 0 for no adjacency

        """ Parameters for filtering clusters """
        max_distance_spread = 0.5
        min_length = 0.05
        max_length = 0.15

        for i, cluster in enumerate(clusters):
            rng = self.left if i == 0 else self.right
            angles = np.linspace(rng[0], rng[1], len(cluster))
            angles = np.radians(angles)

            """Convert to Cartesian coordinates"""
            x = cluster * np.cos(angles)
            y = cluster * np.sin(angles)

            """ Calculate the length of the object that the cluster represents """
            length = (
                np.sqrt((x.max() - x.min()) ** 2 + (y.max() - y.min()) ** 2)
                if len(x) > 1
                else 0
            )

            """ Calculate the mean angle, angular spread, and distance spread of the cluster """
            distance_spread = max(cluster) - min(cluster)
            max_dist = max(cluster)
            max_dist_limit = 1

            if (
                distance_spread <= max_distance_spread
                and min_length <= length <= max_length
                and max_dist <= max_dist_limit
            ):
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
        raw_full = np.array(self.socket.recv_pyobj(), dtype=np.float32)
        raw, term = raw_full[:-1], raw_full[-1]
        obs = np.zeros(self.obs_shape)

        for i, region in enumerate(self.regions):
            start, end = region
            obs[i] = np.min(raw[start:end])

        last = len(self.regions)
        for i in range(self.left[0], self.left[1]):
            j = i - self.left[0]
            obs[last + j] = raw[i]

        last += self.left[1] - self.left[0]
        for i in range(self.right[0], self.right[1]):
            j = i - self.right[0]
            obs[last + j] = raw[i]

        return obs, term

    def no_obs(self) -> np.ndarray:
        return np.full(self.obs_shape, 10.0)

    def reset(self, seed: int = None) -> Tuple[np.ndarray, dict]:
        self.prev_obs = self.no_obs()
        self.time_step = 0
        return self.prev_obs, {}
