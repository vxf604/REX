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
import cam
import landmark_checker

SCALE = 100
arlo = robot.Robot()
landmarkChecker = landmark_checker.LandmarkChecker(landmark_radius=180, scale=SCALE)
cam = cam.Cam()


print("Running ...")
running = True

landmark_detected = False
while running:
    print("Checking for landmark...")
    landmark_detected, c, ids = landmarkChecker.checkForLandmark()
    if landmark_detected:
        print("Landmark detected! Stopping.")
        rvecs, tvecs, objPoints = cam.estimatePose(c)
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
        goal = (0 / SCALE, 2000 / SCALE)
        running = False

cam.stop()
print("Finished")
