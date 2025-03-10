from controller import Supervisor, Robot
import pickle
import numpy as np
import sys


class RobotClient:
    def __init__(self, id: int = 0):
        self.r_path = "/tmp/giorgio_" + str(id) + "_obs"
        self.w_path = "/tmp/giorgio_" + str(id) + "_act"

        self.supervisor = Supervisor()
        self.timestep = int(self.supervisor.getBasicTimeStep())

        self.robot_node = self.supervisor.getFromDef(f"Giorgio_{id}")

        self.lidar = self.supervisor.getDevice("lidar")
        self.lidar.enable(self.timestep)

        self.bumper = self.supervisor.getDevice("bumper")
        self.bumper.enable(self.timestep)

        self.receiver = self.supervisor.getDevice("receiver")
        self.receiver.enable(self.timestep)

        self.left_motor = self.supervisor.getDevice("left wheel motor")
        self.right_motor = self.supervisor.getDevice("right wheel motor")
        self.left_motor.setPosition(float("inf"))
        self.right_motor.setPosition(float("inf"))

        """ Store initial position for resetting """
        self.initial_position = self.robot_node.getField("translation").getSFVec3f()

        self.reset_robot()

    def reset_robot(self):
        """Resets the robot to its initial position."""
        self.robot_node.getField("translation").setSFVec3f(self.initial_position)
        self.supervisor.simulationResetPhysics()

        while self.receiver.getQueueLength() > 0:
            self.receiver.nextPacket

    def run(self):
        while self.supervisor.step(self.timestep) != -1:
            action = self.get_action()
            self.update_motors(action)
            obs = self.read_observation()
            self.send_observation(obs)

            if self.detect_collision():
                self.reset_robot()

            if self.detect_end():
                print("End reached")
                self.reset_robot()

    def get_action(self):
        # with open(self.r_path, "rb") as f:
        #     action = np.array(pickle.load(f))
        action = np.array([1, 0])
        return action

    def send_observation(self, obs):
        pass
        # with open(self.w_path, "wb") as f:
        #     pickle.dump(obs, f)

    def update_motors(self, action):
        # action is pair linear velocity, angular velocity
        # convert to motor speeds
        left_speed = action[0] - action[1]
        right_speed = action[0] + action[1]
        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)

    def read_observation(self):
        return np.clip(np.array(self.lidar.getRangeImage()), 0, 10)

    def detect_collision(self):
        return self.bumper.getValue() == 1

    def detect_end(self):
        return self.receiver.getQueueLength() > 0


if __name__ == "__main__":
    if len(sys.argv) == 0:
        print("Usage: python robot_client.py <robot_id>")
        sys.exit(1)

    client = RobotClient(id=int(sys.argv[1]))
    client.run()
