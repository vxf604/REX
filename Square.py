from time import sleep

import robot

# Create a robot object and initialize
arlo = robot.Robot()
leftSpeed = 65
rightSpeed = 64





print("Running ...")
for i in range (4):
    print(arlo.drive_forward_meter(1))
    print(arlo.go_diff(leftSpeed, rightSpeed, 0, 1))
    sleep(0.5)
    
    
    


# while True:
#     print(arlo.go_diff(leftSpeed, rightSpeed, 0, 1))    

# 2m = 4,9 seconds
# 1m = 2,45 seconds

# send a stop command
print(arlo.stop())
print("Finished")
