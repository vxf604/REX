from time import sleep
import robot

arlo = robot.Robot()

for i in range(5):
    frontSensor = arlo.read_front_ping_sensor()
    print("Front Sensor: ", frontSensor)
    sleep(3)

# 30 cm = 307
# 60 cm = 625
# 90 cm = 953
# 120 cm = 1251
# 150 cm = 1573
# 300 cm = 3106
