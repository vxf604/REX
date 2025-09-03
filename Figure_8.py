from time import sleep

import robot

# Create a robot object and initialize
arlo = robot.Robot()
leftSpeed = 40
rightSpeed = 90

print("Running ...")


print(arlo.go_diff(leftSpeed, rightSpeed, 1, 1))
sleep(6.5)
# print(arlo.go_diff(rightSpeed, leftSpeed, 1, 1))
# sleep(4)

# 2m = 4,9 seconds
# 1m = 2,45 seconds
print("Finished")
