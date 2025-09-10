import robot
import time

arlo = robot.Robot()

print("Running ...")


leftSpeed = 67
rightSpeed = 64
driving = True

while driving:
    print(arlo.go_diff(leftSpeed, rightSpeed, 1, 1))
    leftSensor = arlo.read_left_ping_sensor()
    rightSensor = arlo.read_right_ping_sensor()
    print("Left Sensor: ", leftSensor)

    if frontSensor < 300 or rightSensor < 50 or leftSensor < 50:
        print(arlo.stop())
        if rightSensor > leftSensor:
            print(arlo.go_diff(leftSpeed, rightSpeed, 1, 0))
            while frontSensor < 150:
                frontSensor = arlo.read_front_ping_sensor()
                print("turning left until path clear")
        elif leftSensor > rightSensor:
            print(arlo.go_diff(rightSpeed, leftSpeed, 0, 1))
            while frontSensor < 150:
                frontSensor = arlo.read_front_ping_sensor()
                print("turning right until path clear")
        else:
            print(arlo.go_diff(leftSpeed, rightSpeed, 0, 1))
            while frontSensor < 150:
                frontSensor = arlo.read_front_ping_sensor()
                print("turning left until path clear")
        sleep(0.4)
        print(arlo.go_diff(rightSpeed, leftSpeed, 1, 1))

# print(arlo.go_diff(leftSpeed, rightSpeed, 1, 1))
# while driving:
#     frontSensor = arlo.read_front_ping_sensor()
#     rightSensor = arlo.read_right_ping_sensor()
#     leftSensor = arlo.read_left_ping_sensor()
#     print("Front Sensor: ", frontSensor)
#     print("Right Sensor: ", rightSensor)
#     print("Left Sensor: ", leftSensor)

#     if frontSensor < 300 or rightSensor < 150 or leftSensor < 150:
#         if rightSensor < leftSensor:
#             leftSpeed -= 40
#             rightSpeed += 40
#             print("Turning left")
#         elif leftSensor < rightSensor:
#             rightSpeed -= 40
#             leftSpeed += 40
#             print("Turning right")
#         else:
#             rightSpeed += 40
#             leftSpeed -= 40
#             print("Obstacle in front, Turning right")
#     sleep(0.1)


print("Finished")
