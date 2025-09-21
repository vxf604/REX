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


def checkForLandmark():
    os.makedirs("images", exist_ok=True)

    image = cam.capture_array("main")
    # timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # image_path = os.path.join("images", f"captured_image_{timestamp}.png")
    # cv2.imwrite(image_path, image)

    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()
    corners, ids, rejected = aruco.detectMarkers(
        image, aruco_dict, parameters=parameters
    )
    if ids is None:
        print(" No marker detected!")
        return False, None

    c = corners[0][0]  # first marker detected
    x = int(cv2.norm(c[0] - c[1]))
    Z = f * X / x
    print(f"Distance to landmark Z: {Z} mm")
    cam.stop()
    return True, corners


landmark_detected = False
cam.start(show_preview=False)

while running:
    print(arlo.go_diff(leftSpeed, rightSpeed, 0, 1))
    sleep(0.1)
    print(arlo.stop())
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
        safe_distance = 200
        drive_distance = max (z - safe_distance)
        arlo.go_straight(drive_distance, speed)
        print ("Finished moving to marker.")
        running = False