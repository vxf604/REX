import copy
import math
import os
import random
import cv2
import particle as particle_class
import camera
import numpy as np
import time
from timeit import default_timer as timer
import sys


class Landmark:
    def __init__(self, x, y, color, ID):
        self.x = x
        self.y = y
        self.ID = ID
        self.color = color


# Flags
showGUI = True  # Whether or not to open GUI windows
onRobot = True  # Whether or not we are running on the Arlo robot

if onRobot:
    import robot

    arlo = robot.Robot()


def isRunningOnArlo():
    """Return True if we are running on Arlo, otherwise False.
    You can use this flag to switch the code from running on you laptop to Arlo - you need to do the programming here!
    """
    return onRobot


# Some color constants in BGR format
CRED = (0, 0, 255)
CGREEN = (0, 255, 0)
CBLUE = (255, 0, 0)
CCYAN = (255, 255, 0)
CYELLOW = (0, 255, 255)
CMAGENTA = (255, 0, 255)
CWHITE = (255, 255, 255)
CBLACK = (0, 0, 0)

# Landmarks.
# The robot knows the position of 2 landmarks. Their coordinates are in the unit centimeters [cm].
L1 = Landmark(x=0.0, y=0.0, color=CRED, ID=1)
L2 = Landmark(x=0.0, y=300.0, color=CGREEN, ID=2)
L3 = Landmark(x=400.0, y=0.0, color=CYELLOW, ID=3)
L4 = Landmark(x=400.0, y=300.0, color=CBLUE, ID=4)

landmarks = [L1, L2, L3, L4]

landmarkIDs = {l.ID: l for l in landmarks}

targets = [L2, L3, L4, L1]


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
    offsetY = 50

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
        colour = jet(particle.getWeight() / max_weight)
        cv2.circle(world, (x, y), 2, colour, 2)
        b = (
            int(particle.getX() + 15.0 * np.cos(particle.getTheta())) + offsetX,
            ymax
            - (int(particle.getY() + 15.0 * np.sin(particle.getTheta())) + offsetY),
        )
        cv2.line(world, (x, y), b, colour, 2)

    # Draw landmarks
    for landmark in landmarks:
        ID = landmark.ID
        lm = (int(landmark.x + offsetX), int(ymax - (landmark.y + offsetY)))
        cv2.circle(world, lm, 5, landmark.color, 2)

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
        p = particle_class.Particle(
            600.0 * np.random.ranf() - 100.0,
            600.0 * np.random.ranf() - 250.0,
            np.mod(2.0 * np.pi * np.random.ranf(), 2.0 * np.pi),
            1.0 / num_particles,
        )
        particles.append(p)

    return particles


def roterror(std_rot):
    return random.gauss(0.0, std_rot)


def transerror(std_trans):
    return random.gauss(0.0, std_trans)


def rotation(p, rot, std):
    theta = p.getTheta()
    theta = theta + rot + roterror(std)
    p.setTheta(theta)


def translation1(p, transl1, std):
    x = p.getX()
    y = p.getY()
    theta = p.getTheta()
    d = transl1 + transerror(std)
    x = x + d * np.cos(theta)
    y = y + d * np.sin(theta)
    p.setX(x)
    p.setY(y)


def sample_motion_model(p, rot1, trans):
    if abs(rot1) > 0:
        rotation(p, rot1, 0.10)
        p.setX(p.getX() + transerror(1))
        p.setY(p.getY() + transerror(1))
    elif trans > 0:
        translation1(p, trans, 2)
        p.setTheta(p.getTheta() + roterror(0.05))


def apply_sample_motion_model(particles, rot1, trans):
    for p in particles:
        sample_motion_model(p, rot1, trans)


def apply_motion_from_cmd(particles, cmd):
    if not cmd:
        return
    kind, val = cmd
    if kind == "rotate":
        apply_sample_motion_model(particles, math.radians(val), 0)
    elif kind == "forward":
        apply_sample_motion_model(particles, 0, val)


def sign(x):
    return 1 if x >= 0 else -1


def measurement_model(distance, angle, particle, landmark):
    sigma_d = 15
    sigma_a = 0.15

    x, y = particle.getX(), particle.getY()
    theta = particle.getTheta()
    lx, ly = landmark.x, landmark.y

    d_i = np.sqrt((lx - x) ** 2 + (ly - y) ** 2)

    e_l = np.array([lx - x, ly - y]) / d_i
    e_theta = np.array([math.cos(theta), math.sin(theta)])
    e_theta_hat = np.array([-math.sin(theta), math.cos(theta)])

    fi = sign(np.dot(e_l, e_theta_hat)) * math.acos(np.dot(e_l, e_theta))

    p_distance = (1.0 / (math.sqrt(2.0 * math.pi) * sigma_d)) * math.exp(
        -0.5 * ((distance - d_i) ** 2) / (sigma_d**2)
    )

    p_angle = (1.0 / (math.sqrt(2.0 * math.pi) * sigma_a)) * math.exp(
        -0.5 * ((angle - fi) ** 2) / (sigma_a**2)
    )

    prob = p_angle * p_distance
    return prob


def get_unique_landmarks(objectIDs, dists, angles, landmarkIDs):
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
    return detectedLandmarks, detectedDists, detectedAngles


def angle_to_target(est_pose, target):
    target_x, target_y = target[0], target[1]
    robot_x, robot_y, robot_theta = (
        est_pose.getX(),
        est_pose.getY(),
        est_pose.getTheta(),
    )

    dx = target_x - robot_x
    dy = target_y - robot_y

    t = np.array([dx, dy])
    t = t / np.linalg.norm(t)
    v = np.array([math.cos(robot_theta), math.sin(robot_theta)])

    dot = np.dot(t, v)
    cross = v[0] * t[1] - v[1] * t[0]
    fi = math.acos(dot) * sign(cross)
    return math.degrees(fi)


def distance_to_target(est_pose, target):

    target_x, target_y = target[0], target[1]

    robot_x, robot_y, robot_theta = (
        est_pose.getX(),
        est_pose.getY(),
        est_pose.getTheta(),
    )

    dx = target_x - robot_x
    dy = target_y - robot_y
    distance = math.sqrt(dx**2 + dy**2)

    return distance


def execute_cmd(arlo, cmd):
    if not cmd:
        return
    movement, val = cmd
    if movement == "rotate":
        arlo.rotate_robot(val * -1)
        time.sleep(0.5)
    elif movement == "forward":
        arlo.drive_forward_meter(val / 100.0)
        print(f" Driving forward {val} cm")
        time.sleep(0.5)
    elif movement == "stop":
        arlo.stop()


def motor_control(state, est_pose, targets, seeing, seen2Landmarks):
    target = (targets[0].x, targets[0].y)
    if state == "searching":
        if seen2Landmarks:
            return (None, 0), "rotating"
        return ("rotate", 20.0), "searching"
    print(
        f"est_pose: x: {est_pose.getX()}, y: {est_pose.getY()}, theta: {math.degrees(est_pose.getTheta())}"
    )
    fi = angle_to_target(est_pose, target)
    d = distance_to_target(est_pose, target)

    bearing = math.degrees(
        math.atan2(target[1] - est_pose.getY(), target[0] - est_pose.getX())
    )
    heading = math.degrees(est_pose.getTheta())
    fi = angle_to_target(est_pose, target)  # your function
    print(f"bearing={bearing:.1f}°, heading={heading:.1f}°, fi={fi:.1f}°")
    align_ok = 4

    if state == "rotating":
        # step = max(8.0, min(abs(fi), 35.0))
        # turn = step if fi >= 0 else -step
        next_state = "forward" if abs(fi) < align_ok else "rotating"
        return ("rotate", fi), next_state

    if state == "forward":
        if d < 40:
            return ("rotate", fi), "finish_driving"

        if abs(fi) > align_ok:
            return ("rotate", fi), "forward"
        return ("forward", min(d, 40.0)), "forward"

    if state == "finish_driving":
        return ("forward", d), "reached_target"

    if state == "reached_target":
        if len(targets) > 0:
            targets.pop(0)
            return ("rotate", 20), "searching"
        return ("stop", None), "reached_target"


# Main program #
try:
    if showGUI:
        # Open windows
        if not onRobot:
            WIN_RF1 = "Robot view"
            cv2.namedWindow(WIN_RF1, cv2.WINDOW_NORMAL)

        WIN_World = "World view"

        #
        cv2.namedWindow(WIN_World, cv2.WINDOW_NORMAL)

        cv2.resizeWindow(WIN_World, 520, 520)

        cv2.moveWindow(WIN_World, 720, 50)
    if isRunningOnArlo():
        arlo = robot.Robot()
    # Initialize particles
    num_particles = 3000
    particles = initialize_particles(num_particles)
    state = "searching"
    est_pose = particle_class.estimate_pose(
        particles
    )  # The estimate of the robots current pose

    # Initialize the robot (XXX: You do this)
    if isRunningOnArlo():
        arlo = robot.Robot()

    # Allocate space for world map
    world = np.zeros((700, 700, 3), dtype=np.uint8)

    # Draw map
    draw_world(est_pose, particles, world)
    cv2.imshow(WIN_World, world)

    print("Opening and initializing camera")
    if isRunningOnArlo():
        # cam = camera.Camera(0, robottype='arlo', useCaptureThread=True)
        cam = camera.Camera(0, robottype="arlo", useCaptureThread=False)
    else:
        # cam = camera.Camera(0, robottype='macbookpro', useCaptureThread=True)
        cam = camera.Camera(0, robottype="macbookpro", useCaptureThread=False)

    landmarks_seen = set()
    targetReached = True
    seeing = False
    seen2Landmarks = False
    while True:
        # Fetch next frame
        colour = cam.get_next_frame()
        print("state: ", state)
        # Detect objects
        objectIDs, dists, angles = cam.detect_aruco_objects(colour)
        if not isinstance(objectIDs, type(None)):

            objectIDs, dists, angles = get_unique_landmarks(
                objectIDs, dists, angles, landmarkIDs
            )
            for i in range(len(objectIDs)):
                print(
                    "Object ID = ",
                    objectIDs[i],
                    ", Distance = ",
                    dists[i],
                    ", angle = ",
                    angles[i],
                )
            for landmark in objectIDs:
                if landmark not in landmarks_seen:
                    landmarks_seen.add(landmark)

            # Compute particle weights
            # XXX: You do this
            weightSum = 0.0

            for particle in particles:
                likelyhood = 1.0
                for i in range(len(objectIDs)):
                    likelyhood *= measurement_model(
                        dists[i], angles[i], particle, landmarkIDs[objectIDs[i]]
                    )

                particle.setWeight(likelyhood)
                weightSum += likelyhood

            # Resampling
            # XXX: You do this
            weights = np.zeros((num_particles,))
            for i in range(num_particles):
                if weightSum > 0.0:
                    particles[i].setWeight(particles[i].getWeight() / weightSum)
                else:
                    particles[i].setWeight(1.0 / num_particles)

                weights[i] = particles[i].getWeight()

            particle_indexes = np.random.choice(
                np.arange(num_particles), num_particles, replace=True, p=weights
            )
            new_particles = []

            for i in range(num_particles):
                new_particles.append(copy.copy(particles[particle_indexes[i]]))

            particles = new_particles

            # Draw detected objects
            cam.draw_aruco_objects(colour)
            seeing = len(objectIDs) >= 1

        else:
            # No observation - reset weights to uniform distribution
            seeing = False
            for p in particles:
                p.setWeight(1.0 / num_particles)

        est_pose = particle_class.estimate_pose(particles)

        seen2Landmarks = len(landmarks_seen) >= 4
        if onRobot:
            cmd, state = motor_control(state, est_pose, targets, seeing, seen2Landmarks)
            execute_cmd(arlo, cmd)
            apply_motion_from_cmd(particles, cmd)
        else:
            apply_sample_motion_model(particles, 0, 0)

        if state == "forward":
            landmarks_seen.clear()

        if showGUI:
            # Tegn verden hver gang før visning
            draw_world(est_pose, particles, world)

            # Vis
            if not onRobot:
                cv2.imshow(WIN_RF1, colour)
            cv2.imshow(WIN_World, world)

            if onRobot:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
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
