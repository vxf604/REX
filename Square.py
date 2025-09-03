from time import sleep

import robot

# Create a robot object and initialize
arlo = robot.Robot()
leftSpeed = 68
rightSpeed = 64





print("Running ...")
# for i in range (4):
#     print(arlo.drive_forward_meter(1, leftSpeed, rightSpeed))
#     print(arlo.go_diff(leftSpeed, rightSpeed, 0, 1))
#     sleep(0.8125)
print(arlo.go_diff(leftSpeed, rightSpeed, 0, 1))
sleep(0.8125)

    
    
# 3.61 - 6.86 = 3.25
# 90 degree = 0.8125


 

# 2m = 4,9 seconds
# 1m = 2,45 seconds

# send a stop command
print(arlo.stop())
print("Finished")
