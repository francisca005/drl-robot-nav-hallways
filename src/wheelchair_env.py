import gymnasium as gym
import numpy as np
import socket
import pickle


class WheelchairEnv(gym.Env):
    """
    Custom gymnasium env for Webots robot navigation.
    """

    def __init__(self, conn: socket.socket):
        super(WheelchairEnv, self).__init__()
        self.socket = conn

        """ 
        Action and state space definition.
        Robot will be able to control speed of left and right wheels between 0 and 5.
        State is a vector of 360 lidar readings.
        """
        self.num_lidar_points = 360
        self.action_space = gym.spaces.Box(low=0, high=5, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=0, high=10, shape=(self.num_lidar_points,), dtype=np.float32
        )

    def step(self, action):
        """Sends action to Webots, receives next state, reward, and done flag."""
        try:
            # Send action to Webots robot
            self.socket.sendall(pickle.dumps(action))

            # Receive state, reward, done flag
            data = self.socket.recv(4096)
            obs, reward, done = pickle.loads(data)
            return np.array(obs), reward, done, {}

        except Exception as e:
            print(f"Error in step(): {e}")
            return np.zeros(self.observation_space.shape), 0, True, {}

    def reset(self):
        """Resets the environment (Webots will handle reset)."""
        data = self.socket.recv(4096)
        obs, _, _ = pickle.loads(data)
        return np.array(obs)

    def close(self):
        """Closes the socket connection."""
        if self.socket:
            self.socket.close()
