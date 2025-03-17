import gymnasium as gym
import numpy as np
from sklearn.cluster import DBSCAN
import zmq


class WheelchairEnv(gym.Env):
    def __init__(self, env_id: int):
        super(WheelchairEnv, self).__init__()

        context = zmq.Context()
        self.socket = context.socket(zmq.REQ)
        self.socket.bind("ipc:///tmp/rl/giorgio_" + str(env_id))

        self.env_id = env_id
        self.r_path = "/tmp/giorgio_" + str(env_id) + "_act"
        self.w_path = "/tmp/giorgio_" + str(env_id) + "_obs"

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

    def step(self, action):
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

    def get_reward(self, obs, action):
        v, w = action

        # Reward for moving forward
        r_distance = 1 if v > 0 else -1

        # Collision penalty (if min_range is below a threshold)
        min_range = np.min(obs)
        min_threshold = 0.3
        r_collision = -np.exp(10 * min_range) if min_range < min_threshold else 0

        r_direction = self.direction_reward(obs, action)

        reward = r_distance + r_collision + r_direction
        reward += self.dual_reward(obs, action)

        return reward

    def direction_reward(self, obs, action):
        left = np.array(obs[90:178])
        front = np.array(obs[178:183])
        right = np.array(obs[183:270])

        left_max = np.max(left)
        front_max = np.max(front)
        right_max = np.max(right)

        """ Check which direction has the most free space and reward movement in that direction """
        if max(left_max, front_max, right_max) == front_max:
            return 2 if action[0] > 0 else -2
        elif max(left_max, front_max, right_max) == left_max:
            return 2 if action[1] > 0 else -2

        return 2 if action[1] < 0 else -2

    def collision_reward(self):
        return -30

    def goal_reward(self):
        # maybe the agent shouldn't receive a reward for reaching the end of the corridor
        # because it will be telling the agent that the observation just before the goal is good
        # when in fact it is not
        return 0

    def dual_reward(self, obs, action) -> int:
        # clusters = self.cluster_lidar_readings(obs)
        # n = len(clusters)

        # print("-" * 20)
        # for cluster in clusters:
        #     print(cluster)

        # print("-" * 20)

        return 0

    def cluster_lidar_readings(self, lidar, fov=360):
        """
        Clusters LIDAR readings based on spatial proximity.

        :param lidar_readings: NumPy array of shape (360,) with distance values.
        :param fov: Field of view (degrees), typically 360 for each robot.
        :return: List of clusters (each cluster is a list of indices in lidar_readings).
        """
        n_rays = len(lidar)
        angles = np.linspace(-fov / 2, fov / 2, n_rays)

        # Convert polar to Cartesian coordinates
        x = lidar * np.cos(np.radians(angles))
        y = lidar * np.sin(np.radians(angles))
        points = np.column_stack((x, y))

        # Cluster points using dbscan
        dbscan = DBSCAN(eps=0.1, min_samples=3)
        labels = dbscan.fit_predict(points)

        # Extract clusters
        clusters = {}
        for i, label in enumerate(labels):
            if label == -1:
                continue  # Ignore noise points
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(i)

        return list(clusters.values())

    def send_action_get_obs(self, action) -> np.ndarray:
        self.socket.send_pyobj(action)
        return self.get_observation()

    def get_observation(self) -> np.ndarray:
        return np.array(self.socket.recv_pyobj(), dtype=np.float32)

    def no_obs(self):
        return np.full(self.obs_shape, 10.0)

    def reset(self, seed: int = None):
        self.prev_obs = self.no_obs()
        self.time_step = 0
        return self.prev_obs, {}
