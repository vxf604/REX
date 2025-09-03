from time import sleep

import robot

# Create a robot object and initialize
arlo = robot.Robot()
leftSpeed = 67
rightSpeed = 64

print("Running ...")


print(arlo.go_diff(leftSpeed, rightSpeed, 0, 0))
sleep(2)




# 2m = 4,9 seconds
# 1m = 2,45 seconds
print("Finished")
