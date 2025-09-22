import cv2
import cv2.aruco as aruco
import time
from pprint import *
import os
import datetime
import sys
from time import sleep
import robot
import numpy as np
import matplotlib.pyplot as plt

try:
    import picamera2  # type: ignore

    print("Camera.py: Using picamera2 module")
except ImportError:
    print("Camera.py: picamera2 module not available")
    exit(-1)

arlo = robot.Robot()
speed = 40
leftSpeed = speed + 3
rightSpeed = speed
f = 1226.11  # pixels
X = 145  # mm
cWidth = 1640
cHeight = 1232
imageSize = (cWidth, cHeight)

intrinsic_matrix = np.array([[f, 0, cWidth / 2], [0, f, cHeight / 2], [0, 0, 1]])
distortion_coeffs = np.zeros((5, 1))

FPS = 30
cam = picamera2.Picamera2()
frame_duration_limit = int(1 / FPS * 1000000)  # Microseconds
# Change configuration to set resolution, framerate
picam2_config = cam.create_video_configuration(
    {"size": imageSize, "format": "RGB888"},
    controls={
        "FrameDurationLimits": (frame_duration_limit, frame_duration_limit),
        "ScalerCrop": (0, 0, 3280, 2464),
    },
    queue=False,
)
cam.configure(picam2_config)


print("Running ...")
running = True


fx = 1226.11
fy = 1226.11
cx = 1640 / 2
cy = 1232 / 2

cameraMatrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
distCoeffs = np.zeros((5, 1))


def mapLandmarks():
    image = cam.capture_array("main")
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()
    corners, ids, rejected = aruco.detectMarkers(
        image, aruco_dict, parameters=parameters
    )
    
    if ids is None: 
        print("No markers detected.")
        return []
    
    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
        corners, X, intrinsic_matrix, distortion_coeffs
    ) 
    
    landmark_map = []
    
    
    
    
    for i in range (len(ids)):
        aruco_id = int(ids[i][0])
        tvec = tvecs[i][0]
        landmark_map.append((aruco_id, tvec))
        print (f"ID: {aruco_id} and tvec: {tvec}")
        
    return landmark_map
    
    
while running:
    print(arlo.go_diff(leftSpeed, rightSpeed, 0, 1))
    sleep(0.1)
    print(arlo.stop())
    print("Checking for landmark...")
    landmark_detected, c = checkForLandmark()
    if landmark_detected:
        print("Landmark detected! Stopping.")
        print(arlo.stop())
        rvecs, tvecs, objPoints = cv2.aruco.estimatePoseSingleMarkers(
            c,
            X,
            intrinsic_matrix,
            distortion_coeffs,
        )

        x = tvecs[0][0][0]
        z = tvecs[0][0][2]
        angle_rad = np.arctan2(x, z)
        angle_deg = np.degrees(angle_rad)
        arlo.rotate_robot(angle_deg)
        print(f"Turn {angle_deg:.2f} degrees to face the marker.")
        
        
        distance = arlo.drive_forward_meter((z / 1000) / 4, 64, 67)

cam.stop()
print("Finished")


landmarks = mapLandmarks()  
for aruco_id, tvec in landmarks:
    plt.scatter(tvec[0], tvec[2], label=f"ID {aruco_id}")
plt.xlabel("x (mm)")
plt.ylabel("z (mm)")
plt.legend()
plt.title("Landmark Map in Camera Coordinates")
plt.show()   
        
