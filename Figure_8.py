from time import sleep

import robot

# Create a robot object and initialize
arlo = robot.Robot()
leftSpeed = 43
rightSpeed = 90

print("Running ...")


# print(arlo.go_diff(leftSpeed, rightSpeed, 1, 1))
# sleep(5)
# print(arlo.go_diff(rightSpeed, leftSpeed, 1, 1))
# print(arlo.go_diff(rightSpeed, leftSpeed, 1, 1))
# sleep(8)
# print(arlo.go_diff(rightSpeed, leftSpeed, 1, 1))
# while True:
#     print(arlo.go_diff(leftSpeed, rightSpeed, 1, 1))
#     leftspeed += 5
#     sleep(1)
#     if leftSpeed >= 100:
#         break
    
   
    



loops = 0
while loops < 2:
    print(arlo.go_diff(leftSpeed, rightSpeed, 1, 1))
    sleep(6.85)
    print(arlo.go_diff(rightSpeed, leftSpeed, 1, 1))
    sleep(8)
    loops += 1

# 2m = 4,9 seconds
# 1m = 2,45 seconds
print("Finished")
