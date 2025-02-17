from controller import Supervisor
import socket
import pickle

HOST = "127.0.0.1"
PORT = 5000
EPISODE_LENGTH = 500  # Maximum steps before reset


class RobotClient:
    def __init__(self):
        self.supervisor = Supervisor()  # Use Supervisor instead of Robot
        self.timestep = int(self.supervisor.getBasicTimeStep())

        # Get the robot node
        self.robot_node = self.supervisor.getFromDef(
            "ROBOT"
        )  # Change "ROBOT" to your robot's DEF name

        # Connect to PPO server
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((HOST, PORT))

        # Get sensors
        self.lidar = self.supervisor.getDevice("lidar")
        self.lidar.enable(self.timestep)

        # Get motors
        self.left_motor = self.supervisor.getDevice("left_wheel_motor")
        self.right_motor = self.supervisor.getDevice("right_wheel_motor")
        self.left_motor.setPosition(float("inf"))
        self.right_motor.setPosition(float("inf"))

        # Store initial position for resetting
        self.initial_position = self.robot_node.getField("translation").getSFVec3f()

    def reset_robot(self):
        """Resets the robot to its initial position."""
        self.robot_node.getField("translation").setSFVec3f(self.initial_position)
        self.supervisor.simulationResetPhysics()  # Reset physics to avoid weird movement

    def run(self):
        for episode in range(1000):  # Run multiple episodes
            done = False
            step_count = 0

            while not done and self.supervisor.step(self.timestep) != -1:
                # Get LiDAR readings
                obs = self.lidar.getRangeImage()

                # Define reward function (adjust this for better training)
                reward = (
                    1.0 if sum(obs) > 5 else -1.0
                )  # Example: encourage moving forward

                # End episode if max steps reached or collision detected
                step_count += 1
                if step_count >= EPISODE_LENGTH or self.detect_collision():
                    done = True

                # Send observations, done flag, and reward
                self.socket.sendall(pickle.dumps((obs, done, reward)))

                # Receive action
                data = self.socket.recv(4096)
                action = pickle.loads(data)

                # Apply action
                speed_left, speed_right = action
                self.left_motor.setVelocity(speed_left)
                self.right_motor.setVelocity(speed_right)

            print(f"Episode {episode} finished. Resetting...")
            self.reset_robot()  # Reset robot position after each episode

    def detect_collision(self):
        """Detects if the robot has crashed by checking LiDAR readings."""
        min_distance = min(self.lidar.getRangeImage())
        return min_distance < 0.1  # Adjust threshold based on your environment


if __name__ == "__main__":
    client = RobotClient()
    client.run()
