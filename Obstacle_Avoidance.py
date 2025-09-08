from time import sleep
import robot

# Create a robot object and initialize
arlo = robot.Robot()

print("Running ...")


# send a go_diff command to drive forward
leftSpeed = 67
rightSpeed = 64
driving = True

# print(arlo.go_diff(leftSpeed, rightSpeed, 1, 1))
# while driving:
#     frontSensor = arlo.read_front_ping_sensor()
#     rightSensor = arlo.read_right_ping_sensor()
#     leftSensor = arlo.read_left_ping_sensor()
#     print("Front Sensor: ", frontSensor)
#     print("Right Sensor: ", rightSensor)
#     print("Left Sensor: ", leftSensor)

#     if frontSensor < 300 or rightSensor < 100 or leftSensor < 100:
#         print(arlo.stop())
#         if rightSensor > leftSensor:
#             print(arlo.go_diff(leftSpeed, rightSpeed, 1, 0))
#         elif leftSensor > rightSensor:
#             print(arlo.go_diff(rightSpeed, leftSpeed, 0, 1))
#         else:
#             print(arlo.go_diff(leftSpeed, rightSpeed, 0, 1))
#         sleep(0.4)
#         print(arlo.go_diff(rightSpeed, leftSpeed, 1, 1))

print(arlo.go_diff(leftSpeed, rightSpeed, 1, 1))
while driving:
    frontSensor = arlo.read_front_ping_sensor()
    rightSensor = arlo.read_right_ping_sensor()
    leftSensor = arlo.read_left_ping_sensor()
    print("Front Sensor: ", frontSensor)
    print("Right Sensor: ", rightSensor)
    print("Left Sensor: ", leftSensor)

    if frontSensor < 300 or rightSensor < 100 or leftSensor < 100:
        if rightSensor > leftSensor:
            leftSpeed += 5
            rightSpeed -= 5
        elif leftSensor > rightSensor:
            rightSpeed += 5
            leftSpeed -= 5
        else:
            rightSpeed += 5
            leftSpeed -= 5
    sleep(0.1)


print("Finished")
