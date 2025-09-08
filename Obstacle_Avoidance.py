import robot

# Create a robot object and initialize
arlo = robot.Robot()

print("Running ...")


# send a go_diff command to drive forward
leftSpeed = 67
rightSpeed = 64
driving = True

while driving:
    print(arlo.go_diff(leftSpeed, rightSpeed, 1, 1))
    distance = arlo.read_front_ping_sensor()
    print("Distance: ", distance)
    if distance < 500:
        print(arlo.stop())
        driving = False
        

    

print("Finished")