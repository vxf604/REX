from time import sleep

import robot

# Create a robot object and initialize
arlo = robot.Robot()
leftSpeed = 43
rightSpeed = 90

print("Running ...")
loops = 0
while loops < 3:
    print(arlo.go_diff(leftSpeed, rightSpeed, 1, 1))
    sleep(7)
    print(arlo.go_diff(rightSpeed, leftSpeed - 3, 1, 1))
    sleep(8)
    loops += 1

# 2m = 4,9 seconds
# 1m = 2,45 seconds
print("Finished")
