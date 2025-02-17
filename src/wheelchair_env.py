import gymnasium as gym
import numpy as np
import socket
import pickle


class WheelchairEnv(gym.Env):
    """Custom Gym environment for Webots wheelchair navigation using PPO."""

    def __init__(self, id, host="127.0.0.1", port=5000):
        super(WheelchairEnv, self).__init__()

        self.robot_id = id  # Unique ID for each robot
        self.host = host
        self.port = port + id  # Each robot has a unique port
        self.socket = None

        # Webots settings
        self.num_lidar_points = 360  # Assume LiDAR has 360 readings
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(2,), dtype=np.float32
        )  # Left & Right wheel speeds
        self.observation_space = gym.spaces.Box(
            low=0, high=10, shape=(self.num_lidar_points,), dtype=np.float32
        )  # LiDAR readings

        self.connect_to_robot()  # Establish socket connection

    def connect_to_robot(self):
        """Connect to Webots robot via socket."""
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.host, self.port))
        print(f"Robot {self.robot_id} connected on port {self.port}")

    def step(self, action):
        """Sends action to Webots, receives next state, reward, and done flag."""
        try:
            # Send action to Webots robot
            self.socket.sendall(pickle.dumps(action))

            # Receive state, reward, done flag
            data = self.socket.recv(4096)
            obs, done, reward = pickle.loads(data)
            return np.array(obs), reward, done, {}

        except Exception as e:
            print(f"Error in step(): {e}")
            return np.zeros(self.observation_space.shape), 0, True, {}

    def reset(self):
        """Resets the environment (Webots will handle reset)."""
        return np.zeros(self.observation_space.shape)

    def close(self):
        """Closes the socket connection."""
        if self.socket:
            self.socket.close()
