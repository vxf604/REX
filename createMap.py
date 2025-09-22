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
cam.start(show_preview=False)

cameraMatrix = np.array([[f, 0, cWidth], [0, f, cHeight], [0, 0, 1]], dtype=np.float32)
distCoeffs = np.zeros((5, 1))


def createMap():
    image = cam.capture_array("main")
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()
    corners, ids, rejected = aruco.detectMarkers(
        image, aruco_dict, parameters=parameters
    )
    if ids is None:
        print(" No markers detected!")
        return False

    rvecs, tvecs, objPoints = cv2.aruco.estimatePoseSingleMarkers(
        corners,
        X,
        intrinsic_matrix,
        distortion_coeffs,
    )
    landmarks = []

    for i in range(len(ids)):
        x = tvecs[i][0][0]
        y = tvecs[i][0][1]
        landmarks.append((ids[i][0], x, y))

    plt.scatter([l[1] for l in landmarks], [l[2] for l in landmarks])
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.title("Map of landmarks")
    plt.savefig("landmark_map.png")


print("Running ...")
createMap()
