from time import sleep
import robot

arlo = robot.Robot()

while True:
    frontSensor = arlo.read_front_ping_sensor()
    print("Front Sensor: ", frontSensor)
    sleep(0.5)
