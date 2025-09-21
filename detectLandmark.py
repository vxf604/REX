import cv2
import cv2.aruco as aruco
import time, os, datetime, sys
from time import sleep
from robot import Robot

try:
    import picamera2  
    print("Camera.py: Using picamera2 module")
except ImportError:
    print("Camera.py: picamera2 module not available")
    sys.exit(-1)

arlo = Robot()
leftSpeed = 67
rightSpeed = 64
f = 1226.11  
X = 145      

imageSize = (1640, 1232)
FPS = 30
cam = picamera2.Picamera2()
frame_duration_limit = int(1 / FPS * 1000000) 
picam2_config = cam.create_video_configuration(
    {"size": imageSize, "format": "RGB888"},
    controls={
        "FrameDurationLimits": (frame_duration_limit, frame_duration_limit),
        "ScalerCrop": (0, 0, 3280, 2464),
    },
    queue=False,
)

print("Running ...")

cam.configure(picam2_config)
cam.start(show_preview=False)
time.sleep(1)  

aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
parameters = aruco.DetectorParameters_create()

found = False
while not found:
    frame = cam.capture_array("main")
    corners, ids, _ = aruco.detectMarkers(frame, aruco_dict, parameters=parameters)
    if ids is not None:
        print("Found landmark:", ids)
        arlo.stop()
        found = True
    else:
        arlo.go_diff(leftSpeed, rightSpeed, 0, 1)  # spin
        time.sleep(0.1)

cam.stop()
