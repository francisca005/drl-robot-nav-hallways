import csv
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../..", "src"))

import numpy as np
import zmq
from controller import Supervisor
from robot_state import RobotState


class RobotClient(Supervisor):
    def __init__(self, id: int):
        super(RobotClient, self).__init__()

        self.id = id

        context = zmq.Context()
        self.socket = context.socket(zmq.REP)
        self.socket.connect(f"tcp://127.0.0.1:{5550 + id}")

        self.timestep = int(self.getBasicTimeStep())
        self.dt = self.timestep / 1000.0

        self.robot_node = self.getSelf()

        self.lidar = self.getDevice("Lidar")
        self.lidar.enable(self.timestep)

        self.bumper = self.getDevice("Bumper")
        self.bumper.enable(self.timestep)

        self.receiver = self.getDevice("Receiver")
        self.receiver.enable(self.timestep)

        self.emitter = self.getDevice("Emitter")

        self.left_motor = self.getDevice("left wheel motor")
        self.right_motor = self.getDevice("right wheel motor")
        self.left_motor.setPosition(float("inf"))
        self.right_motor.setPosition(float("inf"))

        # Distance between wheels
        self.l = 0.12

        # Store initial pose for resetting and for false-positive filtering.
        self.initial_position = self.robot_node.getField("translation").getSFVec3f()
        self.initial_rotation = self.robot_node.getField("rotation").getSFRotation()

        # Minimum requirements before accepting an end-strip signal.
        self.min_steps_before_goal = 20
        self.min_distance_from_start_for_goal = 2.0

        # Special command sent by the Python environment when an episode ends by timeout.
        self.timeout_reset_command = np.array([-999.0, -999.0], dtype=np.float32)

        self.positions = []
        self.trajectory_rows = []
        self.steps_since_reset = 0
        self.prev_position_3d = None
        self.last_command = np.array([0.0, 0.0], dtype=np.float32)

        self.reset_robot()

    def clear_receiver_queue(self) -> None:
        """Remove all pending receiver packets."""
        while self.receiver.getQueueLength() > 0:
            self.receiver.nextPacket()

    def reset_robot(self, rotate=True) -> None:
        """Reset the robot to its initial position and clear stale receiver messages."""
        self.robot_node.getField("translation").setSFVec3f(self.initial_position)

        if rotate:
            rotation = self.initial_rotation.copy()
            rotation[3] += np.random.uniform(-0.5, 0.5)
            self.robot_node.getField("rotation").setSFRotation(rotation)
        else:
            self.robot_node.getField("rotation").setSFRotation(self.initial_rotation)

        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        self.simulationResetPhysics()

        self.positions = []
        self.trajectory_rows = []
        self.steps_since_reset = 0
        self.prev_position_3d = None
        self.last_command = np.array([0.0, 0.0], dtype=np.float32)

        self.clear_receiver_queue()

    def append_trajectory_sample(self) -> None:
        """Store position, estimated speed and last commanded action for the current step."""
        pos = np.array(
            self.robot_node.getField("translation").getSFVec3f(),
            dtype=np.float32,
        )

        if self.prev_position_3d is None:
            speed_3d = 0.0
            speed_xy = 0.0
        else:
            delta = pos - self.prev_position_3d
            speed_3d = float(np.linalg.norm(delta) / self.dt)
            speed_xy = float(np.linalg.norm(delta[:2]) / self.dt)

        self.prev_position_3d = pos.copy()

        self.positions.append([float(pos[0]), float(pos[1])])

        self.trajectory_rows.append(
            [
                self.steps_since_reset,
                float(pos[0]),
                float(pos[1]),
                float(pos[2]),
                speed_3d,
                speed_xy,
                float(self.last_command[0]),
                float(self.last_command[1]),
            ]
        )

    def run(self) -> None:
        episode_id = 0

        while self.step(self.timestep) != -1:
            self.append_trajectory_sample()
            self.steps_since_reset += 1

            action = self.get_action()

            if action.shape != (2,):
                break

            if np.allclose(action, self.timeout_reset_command):
                self.save_trajectory(episode_id, outcome="timeout")
                episode_id += 1
                self.reset_robot()

                lidar = self.read_observation()
                state = RobotState(
                    lidar=lidar,
                    prev_action=0,
                    collided=False,
                    goal_reached=False,
                )

                self.send_observation(state)
                continue

            self.last_command = action.copy()
            self.update_motors(action)

            lidar = self.read_observation()
            collided = self.detect_collision()
            end = self.detect_end()

            if collided or end:
                outcome = "collision" if collided else "success"
                self.save_trajectory(episode_id, outcome=outcome)
                episode_id += 1
                self.reset_robot()

            state = RobotState(
                lidar=lidar,
                prev_action=0,  # placeholder, will be set in env
                collided=collided,
                goal_reached=end,
            )

            self.send_observation(state)

        print("Simulation ended, resetting robot...")
        self.reset_robot(rotate=False)
        sys.exit(0)

    def save_trajectory(self, episode_id: int, outcome: str) -> None:
        """
        Save the current episode trajectory.

        Columns:
            step: controller step inside the episode
            x, y, z: robot position
            speed_3d: real speed computed from 3D displacement
            speed_xy: planar speed computed from x/y displacement
            command_v, command_w: last commanded linear/angular velocity
            outcome: success, collision, or timeout
        """
        positions_dir = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "data",
            "positions",
            str(self.id),
        )
        os.makedirs(positions_dir, exist_ok=True)

        output_path = os.path.join(positions_dir, f"t_{episode_id}.csv")

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "step",
                    "x",
                    "y",
                    "z",
                    "speed_3d",
                    "speed_xy",
                    "command_v",
                    "command_w",
                    "outcome",
                ]
            )

            for row in self.trajectory_rows:
                writer.writerow(row + [outcome])

    def get_action(self) -> np.ndarray:
        """Read action from the server."""
        return np.array(self.socket.recv_pyobj(), dtype=np.float32)

    def send_observation(self, obs: RobotState) -> None:
        """Send observation to the server."""
        self.socket.send_pyobj(obs)

    def update_motors(self, action: np.ndarray) -> None:
        """
        Action is a pair: linear velocity, angular velocity.
        Convert it to left and right wheel speeds.
        """
        left_speed = action[0] - action[1] * self.l / 2
        right_speed = action[0] + action[1] * self.l / 2

        self.left_motor.setVelocity(left_speed)
        self.right_motor.setVelocity(right_speed)

    def read_observation(self) -> np.ndarray:
        """Read LiDAR and clip values to avoid inf/nan."""
        return np.clip(np.array(self.lidar.getRangeImage()), 0, 10)

    def detect_collision(self) -> bool:
        """Bumper value is 1 if collision is detected, else 0."""
        collided = self.bumper.getValue() == 1

        if collided:
            message = "collision".encode("utf-8")
            self.emitter.send(message)

        return collided

    def distance_from_start(self) -> float:
        """2D distance from the initial position."""
        pos = self.robot_node.getField("translation").getSFVec3f()

        x, y = pos[0], pos[1]
        initial_x, initial_y = self.initial_position[0], self.initial_position[1]

        return float(((x - initial_x) ** 2 + (y - initial_y) ** 2) ** 0.5)

    def detect_end(self) -> bool:
        """
        Detect whether the robot reached the end strip.

        The original logic accepted any receiver message as success:
            receiver.getQueueLength() > 0

        That caused false positives when stale or early messages remained in the
        receiver queue after reset. To make goal detection more robust, an end
        signal is only accepted if:
            1. a receiver message exists;
            2. the robot has been running for a minimum number of steps;
            3. the robot has moved a minimum distance from its initial position.
        """
        has_message = self.receiver.getQueueLength() > 0

        # Always clear pending messages, even if we decide not to accept them.
        self.clear_receiver_queue()

        if not has_message:
            return False

        if self.steps_since_reset < self.min_steps_before_goal:
            return False

        if self.distance_from_start() < self.min_distance_from_start_for_goal:
            return False

        return True


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python robot_client.py <robot_id>")
        sys.exit(1)

    client = RobotClient(id=int(sys.argv[1]))
    client.run()