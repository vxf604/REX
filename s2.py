import copy
import math
import random
import cv2
import particle
import camera
import numpy as np
import time
from timeit import default_timer as timer
import sys


# Flags
showGUI = True  # Whether or not to open GUI windows
onRobot = True  # Whether or not we are running on the Arlo robot


def isRunningOnArlo():
    """Return True if we are running on Arlo, otherwise False.
    You can use this flag to switch the code from running on you laptop to Arlo - you need to do the programming here!
    """
    return onRobot


try:
    import robot

    onRobot = True
    showGUI = False
except ImportError:
    print("selflocalize.py: robot module not present - forcing not running on Arlo!")
    onRobot = False
    showGUI = True


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
landmarkIDs = [6, 2]
landmarks = {
    6: (0.0, 0.0),  # Coordinates for landmark 1
    2: (100.0, 0.0),  # Coordinates for landmark 2
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


def sample_motion_model(rot1, trans, rot2, p: particle):

    theta = p.getTheta()
    x = p.getX()
    y = p.getY()

    sigma_rot = 0.04
    sigma_trans = 15

    sampled_rot1 = rot1 + np.random.normal(0, sigma_rot)
    sampled_trans = trans + np.random.normal(0, sigma_trans)
    sampled_rot2 = rot2 + np.random.normal(0, sigma_rot)

    updated_x = x + sampled_trans * math.cos(theta + sampled_rot1)
    updated_y = y + sampled_trans * math.sin(theta + sampled_rot1)
    updated_theta = theta + sampled_rot1 + sampled_rot2

    return particle.Particle(updated_x, updated_y, updated_theta, p.getWeight())


def p_hit(z, z_true, sigma):
    return math.exp(-0.5 * ((z_true - z) ** 2) / (sigma**2))


def measurement_model(landmark_pos, p, distance, angle):

    sigma_hit_dist = 300.0
    sigma_hit_angle = 0.3

    particle_pos = (p.getX(), p.getY())
    theta = p.getTheta()

    predicted_dist = math.dist(landmark_pos, particle_pos)
    predicted_angle = (
        math.atan2(landmark_pos[1] - particle_pos[1], landmark_pos[0] - particle_pos[0])
        - theta
    )

    p_hit_dist = p_hit(distance, predicted_dist, sigma_hit_dist)
    p_hit_angle = p_hit(angle, predicted_angle, sigma_hit_angle)

    return p_hit_dist * p_hit_angle


# Main program #
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
    num_particles = 1000
    particles = initialize_particles(num_particles)

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

    prev_time = time.time()
    while True:

        # Move the robot according to user input (only for testing)
        action = cv2.waitKey(10)
        if action == ord("q"):  # Quit
            break

        if not isRunningOnArlo():
            if action == ord("w"):  # Forward
                velocity += 4.0
            elif action == ord("s"):  # Backwards
                velocity -= 4.0
            elif action == ord(" "):  # Stop
                velocity = 0.0
                angular_velocity = 0.0
            elif action == ord("a"):  # Left
                angular_velocity += 0.2
            elif action == ord("d"):  # Right
                angular_velocity -= 0.2

        # Use motor controls to update particles
        # XXX: Make the robot drive
        # XXX: You do this

        current_time = time.time()
        dt = current_time - prev_time
        prev_time = current_time

        if isRunningOnArlo():
            # Arlo driving mechanics
            pass
        else:
            rot1 = dt * angular_velocity / 2
            rot2 = dt * angular_velocity / 2
            trans = dt * velocity
            for p in particles:
                p = sample_motion_model(rot1, trans, rot2, p)

        # Fetch next frame
        colour = cam.get_next_frame()

        # Detect objects
        objectIDs, dists, angles = cam.detect_aruco_objects(colour)
        if not isinstance(objectIDs, type(None)):
            uniqueIDs = set(objectIDs)
            detectedLandmarks = []
            detectedDists = []
            detectedAngles = []

            for uid in uniqueIDs:
                indices = [i for i, id in enumerate(objectIDs) if id == uid]
                closest_id = min(indices, key=lambda i: dists[i])
                detectedLandmarks.append(objectIDs[closest_id])
                detectedDists.append(dists[closest_id])
                detectedAngles.append(angles[closest_id])
            objectIDs, dists, angles = detectedLandmarks, detectedDists, detectedAngles

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

            # Compute particle weights
            # XXX: You do this

            for i in range(len(objectIDs)):
                if objectIDs[i] in landmarkIDs:
                    id = objectIDs[i]
                    pos_landmark = landmarks[id]
                    for p in particles:
                        weight = measurement_model(pos_landmark, p, dists[i], angles[i])
                        p.setWeight(p.getWeight() * weight)

            total_weight = sum(p.getWeight() for p in particles)
            if total_weight > 0:
                for p in particles:
                    p.setWeight(p.getWeight() / total_weight)
            else:
                for p in particles:
                    p.setWeight(1.0 / len(particles))

            # Resampling
            # XXX: You do this
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

            particles = copy.copy(
                random.choices(
                    particles,
                    weights=[p.getWeight() for p in particles],
                    k=len(particles),
                )
            )
            # particles = copy.copy(systematic_resample(particles))

            # Draw detected objects
            cam.draw_aruco_objects(colour)
        else:
            # No observation - reset weights to uniform distribution
            for p in particles:
                p.setWeight(1.0 / num_particles)

        est_pose = particle.estimate_pose(
            particles
        )  # The estimate of the robots current pose
        print(
            "Estimated pose: x = ",
            est_pose.getX(),
            ", y = ",
            est_pose.getY(),
            ", theta = ",
            est_pose.getTheta(),
        )
        if showGUI:
            # Draw map
            draw_world(est_pose, particles, world)

            # Show frame
            cv2.imshow(WIN_RF1, colour)

            # Show world
            cv2.imshow(WIN_World, world)


finally:
    # Make sure to clean up even if an exception occurred

    # Close all windows
    cv2.destroyAllWindows()

    # Clean-up capture thread
    cam.terminateCaptureThread()
