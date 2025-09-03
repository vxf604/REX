from time import sleep

import robot

# Create a robot object and initialize
arlo = robot.Robot()
leftSpeed = 67
rightSpeed = 64

print("Running ...")


# print(arlo.go_diff(leftSpeed, rightSpeed, 0, 0))
# sleep(2)

while True:
    print(arlo.go_diff(leftSpeed, rightSpeed, 1, 1))
    leftspeed += 5
    sleep(1)
    if leftSpeed >= 100:
        break
    
   
    




# 2m = 4,9 seconds
# 1m = 2,45 seconds
print("Finished")
