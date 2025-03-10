from controller import Robot
import time

robot = Robot()
timestep = int(robot.getBasicTimeStep())

touch_sensor = robot.getDevice("touch_sensor")
touch_sensor.enable(timestep)

emitter = robot.getDevice("emitter")

while robot.step(timestep) != -1:
    value = touch_sensor.getValue()

    if value > 0:
        message = "collision".encode("utf-8")
        emitter.send(message)
        time.sleep(0.5)
