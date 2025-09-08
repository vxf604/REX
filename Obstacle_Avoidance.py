from time import sleep
import robot

# Create a robot object and initialize
arlo = robot.Robot()

print("Running ...")


# send a go_diff command to drive forward
leftSpeed = 67
rightSpeed = 64
driving = True

# while driving:
#     print(arlo.go_diff(leftSpeed, rightSpeed, 1, 1))
#     leftSensor = arlo.read_left_ping_sensor()
#     rightSensor = arlo.read_right_ping_sensor()
#     frontSensor = arlo.read_front_ping_sensor()
#     print("Left Sensor: ", leftSensor)
#     print("Right Sensor: ", rightSensor)
#     print("Front Sensor: ", frontSensor)
#     distance = arlo.read_front_ping_sensor()
#     if frontSensor < 200:
#         leftSpeed += 3
#     elif leftSensor > rightSensor:
#         leftSpeed += 1
#         rightSpeed -= 1
#     elif rightSensor > leftSensor:
#         leftSpeed -= 1
#         rightSpeed += 1
print(arlo.go_diff(leftSpeed, rightSpeed, 1, 1))
while driving:
    frontSensor = arlo.read_front_ping_sensor()
    rightSensor = arlo.read_right_ping_sensor()
    leftSensor = arlo.read_left_ping_sensor()
    print("Front Sensor: ", frontSensor)
    print("Right Sensor: ", rightSensor)
    print("Left Sensor: ", leftSensor)

    if frontSensor < 300 or rightSensor < 150 or leftSensor < 150:
        print(arlo.stop())
        if rightSensor > leftSensor:
            print(arlo.go_diff(leftSpeed, rightSpeed, 1, 0))
        elif leftSensor > rightSensor:
            print(arlo.go_diff(rightSpeed, leftSpeed, 0, 1))
        else:
            print(arlo.go_diff(leftSpeed, rightSpeed, 0, 1))
        sleep(0.711)
        print(arlo.go_diff(rightSpeed, leftSpeed, 1, 1))


print("Finished")
