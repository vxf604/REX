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
import random
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
SCALE = 100  # 1 unit = 100 mm
f = 1226.11  # pixels
X = 145  # mm
cWidth = 1640
cHeight = 1232
cx = cWidth / 2
cy = cHeight / 2
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
cam.start(show_preview=False)


print("Running ...")
running = True


fx = 1226.11
fy = 1226.11
cx = 1640 / 2
cy = 1232 / 2
landmark_radius = 180  # mm

cameraMatrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
distCoeffs = np.zeros((5, 1))


def checkForLandmark():
    os.makedirs("images", exist_ok=True)

    image = cam.capture_array("main")
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()
    corners, ids, rejected = aruco.detectMarkers(
        image, aruco_dict, parameters=parameters
    )
    if ids is None:
        print(" No marker detected!")
        return False, None, None

    c = corners[0][0]  # first marker detected
    x = int(cv2.norm(c[0] - c[1]))
    Z = f * X / x
    print(f"Distance to landmark Z: {Z} mm")
    return True, corners, ids


radius_landmark = 1800

landmark_detected = False

while running:
    print(arlo.go_diff(leftSpeed, rightSpeed, 0, 1))
    sleep(0.1)
    print(arlo.stop())
    print("Checking for landmark...")
    landmark_detected, c, ids = checkForLandmark()
    if landmark_detected:
        print("Landmark detected! Stopping.")
        print(arlo.stop())
        rvecs, tvecs, objPoints = cv2.aruco.estimatePoseSingleMarkers(
            c,
            X,
            intrinsic_matrix,
            distortion_coeffs,
        )
        landmarks = []
        id_list = []
        for i in range(len(ids)):
            id = ids[i][0]
            if id in id_list:
                continue

            x = tvecs[i][0][0] / SCALE
            print(f"Landmark ID{id} is {cv2.norm(tvecs[i][0])} mm away from the camera")
            y = tvecs[i][0][2] / SCALE
            landmarks.append((ids[i][0], x, y))
            id_list.append(id)
        goal = (0 / SCALE, 4000 / SCALE)
        path, G = buildRRT(landmarks, goal)
        print("Path:", path)
        save_path_image(landmarks, (0, 0), goal, G, path, filename="rrt_path.png")
        follow_rrt_path(path)
        running = False


cam.stop()
print("Finished")



def rotation1 (p, rot1)
    x, y, theta = p
    theta = (theta + rot1 + roterror()) % (2 * np.pi)
    return (x, y, theta)

def translation1 (p, transl1)
    x, y, theta = p
    transl1 += random.gauss(0, 0.01)
    x = x + transi1 * np.cos(theta +  roterror())
    y = y + transi1 * np.sin(theta + roterror())
    return (x, y, theta)






def sample_motion_model(u, particle):
    rotation = 
    trans = 




def MCL (particle, u, detection, landmarks):
    
    X = []
    for i = 1 in particle: 
        particle_prediction = sample_motion_model (u, particle) 

