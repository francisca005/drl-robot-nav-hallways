from typing import Tuple
import gymnasium as gym
import numpy as np
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
        self.action_space = gym.spaces.Discrete(6)

        v = 3.5
        w = 5
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
        self.time_limit = 20_000

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
            # done = True
            # truncated = True
            # print(f"Time limit reached for robot {self.env_id}")
            pass

        return obs, reward, done, truncated, {}

    def get_reward(self, obs: np.ndarray, action: Tuple[int, int]) -> float:
        v, w = action

        """ Reward for moving forward """
        r_distance = 1 if v > 0 else 0

        """ Collision penalty, exponential on distance if below a threshold """
        min_range = np.min(obs[140:220])
        min_threshold = 1

        if min_range < min_threshold:
            r_collision = -np.exp(3 * (min_threshold - min_range)) + 1
            if self.time_step % 100 == 0:
                print(f"Collision penalty: {r_collision} for robot {self.env_id}")
        else:
            r_collision = 0

        r_direction = self.direction_reward(obs, action)

        reward = r_distance + r_collision + r_direction

        return reward

    def direction_reward(self, obs: np.ndarray, action: Tuple[int, int]) -> int:
        v, w = action

        left = max(obs[150:175])
        front = max(obs[175:185])
        right = max(obs[185:210])

        """ Check which direction has the most free space and reward movement in that direction """
        r = 1
        mx = max(left, front, right)
        threshold = 0.2
        if np.abs(mx - right) < threshold:
            return r if w < 0 else -r
        elif np.abs(mx - left) < threshold:
            return r if w > 0 else -r

        return r if w == 0 else -r

    def collision_reward(self) -> int:
        return -10

    def goal_reward(self) -> int:
        """
        Maybe the agent shouldn't receive a reward for reaching the end of the corridor,
        because it will be telling the agent that the observation just before the goal is good,
        when in fact it is not
        """
        return 0

    def send_action_get_obs(self, action: Tuple[int, int]) -> Tuple[np.ndarray, int]:
        """Send action to server and get observation"""
        self.socket.send_pyobj(action)
        return self.get_observation()

    def get_observation(self) -> Tuple[np.ndarray, int]:
        """Get observation from server"""
        raw = np.array(self.socket.recv_pyobj(), dtype=np.float32)
        obs, term = np.clip(raw[:-1], 0.0, 10.0), int(raw[-1])

        return obs, term

    def no_obs(self) -> np.ndarray:
        return np.full(self.obs_shape, 10.0)

    def reset(self, seed: int = None) -> Tuple[np.ndarray, dict]:
        self.prev_obs = self.no_obs()
        self.time_step = 0
        return self.prev_obs, {}
