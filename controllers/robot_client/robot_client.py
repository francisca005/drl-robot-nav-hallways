from controller import Supervisor
import socket
import pickle
import sys

HOST = "127.0.0.1"
PORT = 5000
EPISODE_LENGTH = 1000


class RobotClient:
    def __init__(self, id: int = 0):
        self.supervisor = Supervisor()
        self.timestep = int(self.supervisor.getBasicTimeStep())

        self.robot_node = self.supervisor.getFromDef(f"Giorgio_{id}")
        print(f"Giorgio {id} connected")

        """ Connect to RL server """
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((HOST, PORT))

        self.lidar = self.supervisor.getDevice("lidar")
        self.lidar.enable(self.timestep)

        self.bumper = self.supervisor.getDevice("bumper")
        self.bumper.enable(self.timestep)

        self.left_motor = self.supervisor.getDevice("left wheel motor")
        self.right_motor = self.supervisor.getDevice("right wheel motor")
        self.left_motor.setPosition(float("inf"))
        self.right_motor.setPosition(float("inf"))

        """ Store initial position for resetting """
        self.initial_position = self.robot_node.getField("translation").getSFVec3f()

    def reset_robot(self):
        """Resets the robot to its initial position."""
        self.robot_node.getField("translation").setSFVec3f(self.initial_position)
        self.supervisor.simulationResetPhysics()

    def run(self):
        for _ in range(1000):  # Run multiple episodes
            done = False
            step_count = 0

            while not done and self.supervisor.step(self.timestep) != -1:
                obs = self.lidar.getRangeImage()

                reward = 0

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

            self.reset_robot()

    def detect_collision(self):
        return self.bumper.getValue() == 1


if __name__ == "__main__":
    if len(sys.argv) == 0:
        print("Usage: python robot_client.py <robot_id>")
        sys.exit(1)

    client = RobotClient(id=int(sys.argv[1]))
    client.run()
