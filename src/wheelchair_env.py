from robot_state import RobotState
from typing import Tuple
import gymnasium as gym
from gymnasium.spaces import Box
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
        State is a vector of 360 lidar readings and the last action taken (as integer).
        """
        self.action_space = gym.spaces.Discrete(6)

        v, w = 1, 3
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

        self.observation_space = Box(
            # 0 to 5 because there are 6 discrete actions
            low=np.concatenate([np.full(360, 0.0), [0]]),
            high=np.concatenate([np.full(360, 10.0), [5]]),
            dtype=np.float64,
        )
        self.obs_shape = self.observation_space.shape

        self.no_obs()
        self.prev_action = 0
        self.prev_pref = 0.0  # Previous preference for side commitment
        self.time_step = 0
        self.time_limit = 20_000

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Takes an action and returns the next observation, reward, done flag, and info.
        :param action: Action to be taken (int)
        :return: Tuple of (observation, reward, done, info)
        """

        self.prev_action = action
        print(f"Entering step function with action: {action}")
        action = self.to_action(action)
        reward = self.get_reward(self.prev_lidar, action)

        obs = self.send_action_get_obs(action)

        if obs.collided:
            reward += self.collision_reward()
        elif obs.goal_reached:
            reward += self.goal_reward()

        self.prev_lidar = obs.lidar
        self.time_step += 1

        print(f"obs: {obs.lidar.shape}, {obs.prev_action}")
        return obs.to_array(), reward, obs.goal_reached, False, {}

    def get_reward(self, obs: np.ndarray, action: Tuple[int, int]) -> float:
        v, _ = action

        """ Reward for moving forward """
        r_distance = 1 if v > 0 else 0

        """ Collision penalty, exponential on distance if below a threshold """
        min_range = np.min(obs[140:220])
        min_threshold = 1

        if min_range < min_threshold:
            r_collision = -np.exp(3 * (min_threshold - min_range)) + 1
        else:
            r_collision = 0

        r_direction = self.direction_reward(obs, action)

        reward = r_distance + r_collision + r_direction

        return reward

    def direction_reward(self, obs: np.ndarray, action: Tuple[int, int]) -> int:
        _, w = action

        left = max(obs[100:175])
        front = max(obs[175:185])
        right = max(obs[185:260])

        """ Check which direction has the most free space and reward movement in that direction """
        r = 1
        mx = max(left, front, right)
        threshold = 0.05
        if np.abs(mx - right) < threshold:
            return r if w < 0 else -r
        elif np.abs(mx - left) < threshold:
            return r if w > 0 else -r

        return r if w == 0 else -r

    def early_side_commitment_reward(
        self, obs: np.ndarray, action: Tuple[int, int]
    ) -> int:
        """
        Reward for early side commitment.
        If the robot is moving forward and has more space on the left, it should turn left.
        If it has more space on the right, it should turn right.
        """
        _, w = action

        front = obs[175:185]
        left = obs[130:175]
        right = obs[185:230]

        threshold = 1.5  # if there is an object closer than this
        r = 2

        if np.any(front < threshold):
            left_clearance = np.mean(left[left > threshold])
            right_clearance = np.mean(right[right > threshold])

            alpha = 0.3
            side_diff = right_clearance - left_clearance
            new_pref = (1 - alpha) * self.prev_pref + alpha * side_diff
            self.prev_pref = new_pref

            if new_pref > 0:
                return r if w < 0 else -r

            return r if w > 0 else -r

        return 0

    def collision_reward(self) -> int:
        return -10

    def goal_reward(self) -> int:
        """
        Maybe the agent shouldn't receive a reward for reaching the end of the corridor,
        because it will be telling the agent that the observation just before the goal is good,
        when in fact it is not
        """
        return 0

    def send_action_get_obs(self, action: Tuple[int, int]) -> RobotState:
        """Send action to server and get observation"""
        self.socket.send_pyobj(action)
        return self.get_observation()

    def get_observation(self) -> RobotState:
        """Get observation from server"""
        state = self.socket.recv_pyobj()
        state.prev_action = self.prev_action

        return state

    def no_obs(self) -> np.ndarray:
        return np.zeros(self.obs_shape, dtype=np.float64)

    def reset(self, seed: int = None) -> Tuple[np.ndarray, dict]:
        obs = self.no_obs()
        self.prev_lidar = obs[:360]
        self.time_step = 0
        return obs, {}
