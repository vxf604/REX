from time import sleep

import robot

# Create a robot object and initialize
arlo = robot.Robot()
leftSpeed = 65
rightSpeed = 64

print("Running ...")
for i in range (4):
    print(arlo.go_diff(leftSpeed, rightSpeed, 1, 1))
    sleep(2.45)
    print(arlo.stop())
    print(arlo.go_diff(leftSpeed, rightSpeed, 0.8, 1))
    sleep(0.5)

    

# 2m = 4,9 seconds
# 1m = 2,45 seconds

# send a stop command
print(arlo.stop())
print("Finished")
