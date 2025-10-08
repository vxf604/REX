import cv2
import cv2.aruco as aruco
from pprint import *
from time import sleep
import robot
import numpy as np
import random
import cam
import landmark_checker
import particle

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

# Sørg for at standard deviation passer med hvad vores x og y er i (mm eller cm eller m)

def roterror (std_rot = 0.01):
    return random.gauss(0.0, std_rot)

def transerror (trans1, std_trans = 0.01):
    return random.gauss(0.0, std_trans)

def rotation1 (p, rot1)
    x, y, theta = p
    theta = (theta + rot1 + roterror()) % (2 * np.pi) - np.pi
    return (x, y, theta)

def rotation2 (p, rot2)
    x, y, theta = p
    theta = (theta + rot2 + roterror()) % (2 * np.pi) - np.pi
    return (x, y, theta)

def translation1 (p, transl1)
    x, y, theta = p
    d = transl1 + transerror(transl1)
    x = x + d * np.cos(theta +  roterror())
    y = y + d * np.sin(theta + roterror())
    return (x, y, theta)


def sample_motion_model(p, rot1, trans, rot2):
    p = rotation1(p, rot1)
    p = rotation2(p, rot2)
    p = translation (p, trans)
    return p



def ini_particles(n):
    particles = []
    for i in range (n):
        x = random.uniform(-2000, 2000) / SCALE
        y = random.uniform(-2000, 2000) / SCALE
        theta = random.uniform(-np.pi, np.pi)
        particles.append ((x, y, theta))
    return particles


def normal_distribution(mu, sigma, x):
    mu = distance [i]
    sigma = 0.1
    x = landmarks[objectIDs[i]][0]
    return (1 / (math.sqrt(2 * math.pi) * sigma)) * math.exp(-0.5 * ((x - mu) / sigma) ** 2)


def predicted_distance (p, landmark):
    lx , ly = landmark
    x, y = p
    return np.sqrt ((lx - x)**2 + (ly - y)**2)



def measurement_model(p, landmarks):
    objectIDs, distance, angles = cam.detect_aruco_landmarks()
    
    if not instance(objectID, type[None]):
        for i in range (len (objectIDs))
        print ("Object ID: ", objectIDs[i], " Distance: ", distance[i], " Angle: ", angles[i])
            predicted_dist = predicted_distance (p, landmarks[objectIDs[i]])
            prob = normal_distribution(predicted_dist, 0.1, distance[i])
            return prob
        
    
        
    
def MCL (particles, control_rtr, detections, LANDMARKS,
             sig_d=10.0, sig_b=math.radians(8.0), angles_deg=True):
    for (x, y, theta, w) in particle:
        new_x, new_y, new_theta = sample_motion_model((x, y, theta), rot1, trans, rot2)
        particle.append((new_x, new_y, new_theta, w))
    
        
    weights = []
    for (x,y,theta,w) = particle:
        weight = measurement_model((x,y,theta), LANDMARKS)
        weights.append(weight)
        
    
        

