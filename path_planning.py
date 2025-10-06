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


def save_path_image(landmarks, start, goal, G, path, filename="rrt_path.png"):
    plt.figure(figsize=(8, 8))

    # Draw all RRT nodes
    if len(G) > 0:
        Gx, Gy = zip(*[(gx * SCALE, gy * SCALE) for gx, gy in G])
        plt.scatter(Gx, Gy, c="lightgray", s=10, label="RRT nodes")

    # Draw landmarks as circles
    for lid, lx, ly in landmarks:
        lx_mm = lx * SCALE
        ly_mm = ly * SCALE
        circle = plt.Circle(
            (lx_mm, ly_mm), landmark_radius, color="red", fill=False, linestyle="--"
        )
        plt.gca().add_patch(circle)
        plt.text(lx_mm, ly_mm, f"ID{lid}", color="red")

    # Draw start and goal
    plt.scatter(
        start[0] * SCALE, start[1] * SCALE, c="green", s=100, marker="o", label="Start"
    )
    plt.scatter(
        goal[0] * SCALE, goal[1] * SCALE, c="blue", s=100, marker="*", label="Goal"
    )

    # Draw the final path
    if path is not None and len(path) > 1:
        px, py = zip(*[(px * SCALE, py * SCALE) for px, py in path])
        plt.plot(px, py, c="black", linewidth=2, label="Path")

    plt.xlabel("X [mm]")
    plt.ylabel("Y [mm]")
    plt.title("RRT Path Planning")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)

    # Save instead of show
    plt.savefig(filename)
    plt.close()
    print(f"Path image saved as {filename}")


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


radius_landmark = 1800  # mm


def distance(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def in_collision(point, landmarks, robot_radius=150):
    x, y = point
    for landmark in landmarks:
        id, map_x, map_y = landmark
        landmark_pos = (map_x, map_y)
        d = distance(point, landmark_pos)
        if d <= (landmark_radius / SCALE) + (robot_radius / SCALE):
            return True
    return False


def randConf():
    return (random.uniform(-2000, 2000) / SCALE, random.uniform(-2000, 2000) / SCALE)


def NEAREST_VERTEX(v, G):
    min = distance(G[0], v)
    minV = G[0]
    for i in range(len(G)):
        d = distance(v, G[i])
        if d < min or i == 0:
            min = d
            minV = G[i]
    return minV


def Steer(q_near, q_rand, delta_q=300 / SCALE):
    dx, dy = q_rand[0] - q_near[0], q_rand[1] - q_near[1]
    d = distance(q_near, q_rand)

    if d < delta_q:
        return q_rand
    else:
        return (q_near[0] + delta_q * dx / d, q_near[1] + delta_q * dy / d)


def buildRRT(landmarks, goal, delta_q=300 / SCALE):
    start = (0, 0)
    G = [start]
    parent = {0: None}

    goal_index = None
    i = 0

    while goal_index is None:
        i += 1
        q_rand = randConf()
        q_near = NEAREST_VERTEX(q_rand, G)
        q_new = Steer(q_near, q_rand, delta_q)

        if in_collision(q_new, landmarks):
            continue

        G.append(q_new)
        parent[len(G) - 1] = G.index(q_near)

        if distance(q_new, goal) < delta_q:
            goal_index = len(G) - 1

    path = []
    node = goal_index
    while node is not None:
        path.append(G[node])
        node = parent[node]
    path.reverse()

    path.append(goal)

    return path, G


current_heading = 0


def follow_rrt_path(path):
    global current_heading
    current_pos = np.array([0.0, 0.0])

    for i in range(1, len(path)):
        start = path[i - 1]
        target = path[i]

        dx = target[0] - start[0]
        dy = target[1] - start[1]

        desired_angle_deg = np.degrees(np.arctan2(dx, dy))
        rotation_needed = desired_angle_deg - current_heading

        print(f"Rotating {rotation_needed:.2f} degrees")
        arlo.rotate_robot(rotation_needed)
        current_heading = desired_angle_deg

        distance_m = np.sqrt(dx**2 + dy**2) * (SCALE / 1000.0)
        print(f"Driving forward {distance_m:.3f} meters")
        arlo.drive_forward_meter(distance_m, 64, 67)

        rad = np.radians(current_heading)
        current_pos += np.array([np.sin(rad), np.cos(rad)]) * distance_m


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
        goal = (0 / SCALE, 1750 / SCALE)
        path, G = buildRRT(landmarks, goal)
        print("Path:", path)
        save_path_image(landmarks, (0, 0), goal, G, path, filename="rrt_path.png")
        follow_rrt_path(path)
        running = False


cam.stop()
print("Finished")
