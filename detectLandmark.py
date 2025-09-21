import cv2
import cv2.aruco as aruco
import time
from pprint import *
import os
import datetime
import sys
from time import sleep
import robot

try:
    import picamera2  # type: ignore

    print("Camera.py: Using picamera2 module")
except ImportError:
    print("Camera.py: picamera2 module not available")
    exit(-1)

arlo = robot.Robot()
leftSpeed = int(67 / 2)
rightSpeed = int(64 / 2)
f = 1226.11  # pixels
X = 145  # mm

imageSize = (1640, 1232)
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


def checkForLandmark():
    os.makedirs("images", exist_ok=True)

    image = cam.capture_array("main")
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = os.path.join("images", f"captured_image_{timestamp}.png")
    cv2.imwrite(image_path, image)

    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()
    corners, ids, rejected = aruco.detectMarkers(
        image, aruco_dict, parameters=parameters
    )
    if ids is None:
        print(" No marker detected!")
        return False

    c = corners[0][0]  # first marker detected
    x = int(cv2.norm(c[0] - c[1]))
    Z = f * X / x
    print(f"Distance to landmark Z: {Z} mm")
    cam.stop()
    return True


turning = False
landmark_detected = False
cam.start(show_preview=False)

while running:

    if not turning:
        print(arlo.go_diff(leftSpeed, rightSpeed, 0, 1))
        turning = True
    landmark_detected = checkForLandmark()
    if landmark_detected:
        print("Landmark detected! Stopping.")
        print(arlo.stop())
        running = False
