import robot

# Create a robot object and initialize
arlo = robot.Robot()

print("Running ...")


# send a go_diff command to drive forward
leftSpeed = 67
rightSpeed = 64
driving = True

while driving:
    print(arlo.go_diff(leftSpeed, rightSpeed, 1, 1))
    leftSensor = arlo.read_left_ping_sensor()
    rightSensor = arlo.read_right_ping_sensor()
    print("Left Sensor: ", leftSensor)
    print("Right Sensor: ", rightSensor)
    distance = arlo.read_front_ping_sensor()
    if leftSensor > rightSensor:
        leftSpeed += 2
        rightSpeed -= 2
    elif rightSensor > leftSensor:
        leftSpeed -= 2
        rightSpeed += 2


print("Finished")
