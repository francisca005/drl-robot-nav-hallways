from controller import Robot


def run():
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    touch_sensor = robot.getDevice("Bumper")
    touch_sensor.enable(timestep)

    emitter = robot.getDevice("Emitter")

    while robot.step(timestep) != -1:
        value = touch_sensor.getValue()

        if value > 0:
            message = "end".encode("utf-8")
            emitter.send(message)


if __name__ == "__main__":
    run()
