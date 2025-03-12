from controller import Robot
import time

robot = Robot()
timestep = int(robot.getBasicTimeStep())

touch_sensor = robot.getDevice("Bumper")
touch_sensor.enable(timestep)

emitter = robot.getDevice("Emitter")
print("End strip controller started")

while robot.step(timestep) != -1:
    value = touch_sensor.getValue()

    if value > 0:
        print("strip detected collision")
        message = "collision".encode("utf-8")
        emitter.send(message)
