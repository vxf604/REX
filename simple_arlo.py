from time import sleep

import robot

# Create a robot object and initialize
arlo = robot.Robot()

print("Running ...")


# send a go_diff command to drive forward
leftSpeed = 65
rightSpeed = 64
print(arlo.go_diff(leftSpeed, rightSpeed, 1, 1))

# Wait a bit while robot moves forward
sleep(2.45)

# 2m = 4,9 seconds
# 1m = 2,45 seconds

# send a stop command
print(arlo.stop())
print("Finished")
