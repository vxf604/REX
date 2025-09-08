import robot

# Create a robot object and initialize
arlo = robot.Robot()

print("Running ...")


# send a go_diff command to drive forward
leftSpeed = 67
rightSpeed = 64
distance = 200

while distance > 100:
    print(arlo.drive_forward_meter(2, leftSpeed, rightSpeed))
    distance = arlo.read_front_ping_sensor
    print("Distance: ", distance)

    

print("Finished")