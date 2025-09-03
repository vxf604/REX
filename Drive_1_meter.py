from time import sleep

import robot

# Create a robot object and initialize
arlo = robot.Robot()

print("Running ...")


# send a go_diff command to drive forward
leftSpeed = 67
rightSpeed = 63
print(arlo.drive_forward_meter(2, leftSpeed, rightSpeed))



# 2m = 4,9 seconds
# 1m = 2,45 seconds
print("Finished")
