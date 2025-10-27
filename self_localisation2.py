import os
import cv2
from pprint import *
from time import sleep
import numpy as np
import random
import camera
import landmark_checker
import particle
import sys
import math
import copy
import time


def isRunningOnArlo():
    """Return True if we are running on Arlo, otherwise False.
    You can use this flag to switch the code from running on you laptop to Arlo - you need to do the programming here!
    """
    return onRobot


try:
    import robot

    showGUI = False
    onRobot = True
except ImportError:
    print("selflocalize.py: robot module not present - forcing not running on Arlo!")
    onRobot = False
    showGUI = True


running = True
# Some color constants in BGR format

CRED = (0, 0, 255)
CGREEN = (0, 255, 0)
CBLUE = (255, 0, 0)
CCYAN = (255, 255, 0)
CYELLOW = (0, 255, 255)
CMAGENTA = (255, 0, 255)
CWHITE = (255, 255, 255)
CBLACK = (0, 0, 0)

landmarkIDs = [5, 6]
landmarks = {
    5: (0.0, 0.0),  # Coordinates for landmark 1 Red
    6: (150.0, 0.0),  # Coordinates for landmark 2 Green
}
landmark_colors = [CRED, CGREEN]  # Colors used when drawing the landmarks


def jet(x):
    """Colour map for drawing particles. This function determines the colour of
    a particle from its weight."""
    r = (
        (x >= 3.0 / 8.0 and x < 5.0 / 8.0) * (4.0 * x - 3.0 / 2.0)
        + (x >= 5.0 / 8.0 and x < 7.0 / 8.0)
        + (x >= 7.0 / 8.0) * (-4.0 * x + 9.0 / 2.0)
    )
    g = (
        (x >= 1.0 / 8.0 and x < 3.0 / 8.0) * (4.0 * x - 1.0 / 2.0)
        + (x >= 3.0 / 8.0 and x < 5.0 / 8.0)
        + (x >= 5.0 / 8.0 and x < 7.0 / 8.0) * (-4.0 * x + 7.0 / 2.0)
    )
    b = (
        (x < 1.0 / 8.0) * (4.0 * x + 1.0 / 2.0)
        + (x >= 1.0 / 8.0 and x < 3.0 / 8.0)
        + (x >= 3.0 / 8.0 and x < 5.0 / 8.0) * (-4.0 * x + 5.0 / 2.0)
    )

    return (255.0 * r, 255.0 * g, 255.0 * b)


def draw_world(est_pose, particles, world):
    """Visualization.
    This functions draws robots position in the world coordinate system."""

    # Fix the origin of the coordinate system
    offsetX = 100
    offsetY = 250

    # Constant needed for transforming from world coordinates to screen coordinates (flip the y-axis)
    ymax = world.shape[0]

    world[:] = CWHITE  # Clear background to white

    # Find largest weight
    max_weight = 0
    for particle in particles:
        max_weight = max(max_weight, particle.getWeight())

    # Draw particles
    for particle in particles:
        x = int(particle.getX() + offsetX)
        y = ymax - (int(particle.getY() + offsetY))
        if max_weight == 0:
            colour = jet(0)
        else:
            colour = jet(particle.getWeight() / max_weight)
        cv2.circle(world, (x, y), 2, colour, 2)
        b = (
            int(particle.getX() + 15.0 * np.cos(particle.getTheta())) + offsetX,
            ymax
            - (int(particle.getY() + 15.0 * np.sin(particle.getTheta())) + offsetY),
        )
        cv2.line(world, (x, y), b, colour, 2)

    # Draw landmarks
    for i in range(len(landmarkIDs)):
        ID = landmarkIDs[i]
        lm = (int(landmarks[ID][0] + offsetX), int(ymax - (landmarks[ID][1] + offsetY)))
        cv2.circle(world, lm, 5, landmark_colors[i], 2)

    # Draw estimated robot pose
    a = (int(est_pose.getX()) + offsetX, ymax - (int(est_pose.getY()) + offsetY))
    b = (
        int(est_pose.getX() + 15.0 * np.cos(est_pose.getTheta())) + offsetX,
        ymax - (int(est_pose.getY() + 15.0 * np.sin(est_pose.getTheta())) + offsetY),
    )
    cv2.circle(world, a, 5, CMAGENTA, 2)
    cv2.line(world, a, b, CMAGENTA, 2)


def initialize_particles(num_particles):
    particles = []
    for i in range(num_particles):
        # Random starting points.
        p = particle.Particle(
            600.0 * np.random.ranf() - 100.0,
            600.0 * np.random.ranf() - 250.0,
            np.mod(2.0 * np.pi * np.random.ranf(), 2.0 * np.pi),
            1.0 / num_particles,
        )
        particles.append(p)

    return particles


# Sørg for at standard deviation passer med hvad vores x og y er i (mm eller cm eller m)
def roterror(std_rot=0.1):
    return random.gauss(0.0, std_rot)


def transerror(std_trans=2.5):
    return random.gauss(0.0, std_trans)


def rotation1(p, rot1):
    theta = p.getTheta()
    theta = theta + rot1 + roterror()
    p.setTheta(theta)
    return p


def rotation2(p, rot2):
    theta = p.getTheta()
    theta = theta + rot2 + roterror()
    p.setTheta(theta)
    return p


def translation1(p, transl1):
    x = p.getX()
    y = p.getY()
    theta = p.getTheta()
    d = transl1 + transerror()
    x = x + d * np.cos(theta)
    y = y + d * np.sin(theta)
    p.setX(x)
    p.setY(y)
    return p


def sample_motion_model(p, rot1, trans, rot2):

    p = rotation1(p, rot1)
    p = translation1(p, trans)
    p = rotation2(p, rot2)

    return p


def normal_distribution(mu, sigma, x):
    return (1 / (math.sqrt(2 * math.pi) * sigma)) * math.exp(
        -0.5 * ((x - mu) / sigma) ** 2
    )


def predicted_distance(p, landmark):
    lx, ly = landmark
    x = p.getX()
    y = p.getY()
    return np.sqrt((lx - x) ** 2 + (ly - y) ** 2)


def predicted_angle(p, landmark):
    lx, ly = landmark
    x, y, theta = p.getX(), p.getY(), p.getTheta()

    e_theta = np.array([math.cos(theta), math.sin(theta)])
    e_theta_hat = np.array([-math.sin(theta), math.cos(theta)])
    e_l = np.array([lx - x, ly - y])
    e_l = e_l / np.linalg.norm(e_l)

    phi = np.sign(np.dot(e_l, e_theta_hat)) * math.acos(
        np.clip(np.dot(e_l, e_theta), -1.0, 1.0)
    )
    return phi


def sign(x):
    return 1 if x >= 0 else -1


def measurement_model(distance, angle, particle, landmark):
    sigma_d = 15
    sigma_a = math.radians(2.0)

    x, y = particle.getX(), particle.getY()
    theta = particle.getTheta()
    lx, ly = landmark

    d_i = np.sqrt((lx - x) ** 2 + (ly - y) ** 2)

    e_l = np.array([lx - x, ly - y]) / d_i
    e_theta = np.array([math.cos(theta), math.sin(theta)])
    e_theta_hat = np.array([-math.sin(theta), math.cos(theta)])

    fi = sign(np.dot(e_l, e_theta_hat)) * math.acos(
        np.clip(np.dot(e_l, e_theta), -1.0, 1.0)
    )

    p_distance = (1.0 / (math.sqrt(2.0 * math.pi) * sigma_d)) * math.exp(
        -0.5 * ((distance - d_i) ** 2) / (sigma_d**2)
    )
    p_angle = (1.0 / (math.sqrt(2.0 * math.pi) * sigma_a)) * math.exp(
        -0.5 * ((angle - fi) ** 2) / (sigma_a**2)
    )

    prob = p_angle * p_distance
    return prob


def systematic_resample(particles):
    N = len(particles)
    positions = (np.arange(N) + random.random()) / N
    cumulative_sum = np.cumsum([p.getWeight() for p in particles])
    cumulative_sum[-1] = 1.0
    i, j = 0, 0
    new_particles = []
    while i < N:
        if positions[i] < cumulative_sum[j]:
            new_particles.append(copy.copy(particles[j]))
            i += 1
        else:
            j += 1
    return new_particles


def particle_filter(particles, objectIDs, dists, angles, colour):
    if not isinstance(objectIDs, type(None)):
        uniqueIDs = set(objectIDs)
        detectedLandmarks = []
        detectedDists = []
        detectedAngles = []

        for uid in uniqueIDs:
            indices = [i for i, id in enumerate(objectIDs) if id == uid]
            closest_id = min(indices, key=lambda i: dists[i])
            if uid in landmarkIDs:
                detectedLandmarks.append(objectIDs[closest_id])
                detectedDists.append(dists[closest_id])
                detectedAngles.append(angles[closest_id])

        objectIDs, dists, angles = (
            detectedLandmarks,
            detectedDists,
            detectedAngles,
        )

        # List detected objects
        for i in range(len(objectIDs)):
            print(
                "Object ID = ",
                objectIDs[i],
                ", Distance = ",
                dists[i],
                ", angle = ",
                angles[i],
            )

        new_particles = []
        p_len = len(particles)
        for j in range(p_len):
            p = particles[j]
            new_p = copy.copy(p)

            # Combine measurements from all visible landmarks
            total_prob = 1.0
            for i in range(len(objectIDs)):
                landmark_id = int(objectIDs[i])
                if landmark_id not in landmarks:
                    continue
                total_prob *= measurement_model(
                    dists[i], angles[i], new_p, landmarks[landmark_id]
                )

            new_p.setWeight(total_prob)
            new_particles.append(new_p)

        total_weight = sum(p.getWeight() for p in new_particles)
        if total_weight > 0:
            for p in new_particles:
                p.setWeight(p.getWeight() / total_weight)
        else:
            for p in new_particles:
                p.setWeight(1.0 / len(new_particles))

        particles = systematic_resample(new_particles)

        # Draw detected objects
        cam.draw_aruco_objects(colour)
    else:
        # No observation - reset weights to uniform distribution
        for p in particles:
            p.setWeight(1.0 / num_particles)

    return particles, objectIDs, dists, angles


def init_robot_localization(particles):
    landmarks_seen = set()
    while len(landmarks_seen) < 2:
        colour = cam.get_next_frame()
        rotated_degrees = math.radians(20)

        arlo.rotate_robot(20)
        sleep(0.6)

        objectIDs, dists, angles = cam.detect_aruco_objects(colour)
        if objectIDs is not None:
            landmarks_seen.update(int(i) for i in objectIDs)

        particles, objectIDs, dists, angles = particle_filter(
            particles, objectIDs, dists, angles, colour
        )

        for i in range(len(particles)):
            particles[i] = sample_motion_model(
                particles[i], math.radians(rotated_degrees), 0.0, 0.0
            )


try:
    if showGUI:
        # Open windows
        WIN_RF1 = "Robot view"
        cv2.namedWindow(WIN_RF1)
        cv2.moveWindow(WIN_RF1, 50, 50)

        WIN_World = "World view"
        cv2.namedWindow(WIN_World)
        cv2.moveWindow(WIN_World, 500, 50)

    # Initialize particles
    num_particles = 2000
    particles = initialize_particles(num_particles)

    # initialise landmarks:
    objectIDs, dists, angles = [], [], []

    est_pose = particle.estimate_pose(
        particles
    )  # The estimate of the robots current pose

    # Driving parameters
    velocity = 0.0  # cm/sec
    angular_velocity = 0.0  # radians/sec

    # Initialize the robot (XXX: You do this)
    if isRunningOnArlo():
        arlo = robot.Robot()

    # Allocate space for world map
    world = np.zeros((500, 500, 3), dtype=np.uint8)

    # Draw map
    draw_world(est_pose, particles, world)

    print("Opening and initializing camera")
    if isRunningOnArlo():
        # cam = camera.Camera(0, robottype='arlo', useCaptureThread=True)
        cam = camera.Camera(0, robottype="arlo", useCaptureThread=False)
    else:
        # cam = camera.Camera(0, robottype='macbookpro', useCaptureThread=True)
        cam = camera.Camera(0, robottype="macbookpro", useCaptureThread=False)
    init = True
    distance_cm = 0.0
    if isRunningOnArlo():
        init_robot_localization(particles)
    target = (
        (landmarks[5][0] + landmarks[6][0]) / 2,
        (landmarks[5][1] + landmarks[6][1]) / 2,
    )

    while True:

        # Move the robot according to user input (only for testing)
        action = cv2.waitKey(10)
        if action == ord("q"):  # Quit
            break

        if not isRunningOnArlo():
            if action == ord("w"):  # Forward
                velocity += 4.0
            elif action == ord("x"):  # Backwards
                velocity -= 4.0
            elif action == ord("s"):  # Stop
                velocity = 0.0
                angular_velocity = 0.0
            elif action == ord("a"):  # Left
                angular_velocity += 0.2
            elif action == ord("d"):  # Right
                angular_velocity -= 0.2

            for i in range(len(particles)):
                particles[i] = sample_motion_model(
                    particles[i], angular_velocity, velocity, 0.0
                )
        else:
            # Arlo controls
            est_pose = particle.estimate_pose(particles)
            print(
                "Estimated pose: x=",
                est_pose.getX(),
                " y=",
                est_pose.getY(),
                " theta=",
                est_pose.getTheta(),
            )

            if (
                abs(est_pose.x - target[0]) < 20.0
                and abs(est_pose.y - target[1]) < 20.0
            ):
                arlo.stop()
                print("Reached target")
                break

            angle_change = 0.0

            dx = target[0] - est_pose.getX()
            dy = target[1] - est_pose.getY()

            theta = est_pose.getTheta()
            t = np.array([dx, dy])
            t = t / np.linalg.norm(t)
            v = np.array([math.cos(theta), math.sin(theta)])

            dot = np.dot(t, v)
            cross = v[0] * t[1] - v[1] * t[0]
            fi = math.acos(dot) * sign(cross)
            target_distance = np.sqrt((dx) ** 2 + (dy) ** 2)
            drive_cm = 30.0
            angle_change = fi
            print("Distance to target: ", distance_cm)
            print(f"Rotating {fi} radians")
            arlo.rotate_robot(math.degrees(fi))
            sleep(0.5)
            arlo.drive_forward_meter(drive_cm / 100.0)

            for i in range(len(particles)):
                particles[i] = sample_motion_model(
                    particles[i], angle_change, drive_cm, 0.0
                )

        # Get new camera image
        colour = cam.get_next_frame()
        objectIDs, dists, angles = cam.detect_aruco_objects(colour)
        particles, objectIDs, dists, angles = particle_filter(
            particles, objectIDs, dists, angles, colour
        )

        # Estimate robot pose
        est_pose = particle.estimate_pose(particles)

        # Visualization
        if showGUI:
            draw_world(est_pose, particles, world)
            cv2.imshow(WIN_RF1, colour)
            cv2.imshow(WIN_World, world)
        else:
            draw_world(est_pose, particles, world)
            folder = "images"
            os.makedirs(folder, exist_ok=True)
            cv2.imwrite(
                os.path.join(folder, f"Current_world_{int(time.time())}.png"), world
            )

finally:
    # Make sure to clean up even if an exception occurred

    # Close all windows
    cv2.destroyAllWindows()

    # Clean-up capture thread
    cam.terminateCaptureThread()
