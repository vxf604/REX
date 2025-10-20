import cv2
import particle
import camera
import numpy as np
import time
from timeit import default_timer as timer
import math
import random
import copy


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

    showGUI = False
    onRobot = True
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
    2: (200.0, 0.0),  # Coordinates for landmark 2
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


# --- Motion Model (sample_motion_model) ---
def sample_motion_model(p, rot1, trans, rot2):
    """Implements p(x_t | x_{t-1}, u_t) using noisy rotation-translation-rotation model"""
    # Add Gaussian noise to rotation and translation
    rot1_hat = rot1 + random.gauss(0.0, math.radians(2))
    trans_hat = trans + random.gauss(0.0, abs(trans) * 0.05)
    rot2_hat = rot2 + random.gauss(0.0, math.radians(2))

    # Update particle pose
    x = p.getX() + trans_hat * math.cos(p.getTheta() + rot1_hat)
    y = p.getY() + trans_hat * math.sin(p.getTheta() + rot1_hat)
    theta = (p.getTheta() + rot1_hat + rot2_hat) % (2 * math.pi)

    p.setX(x)
    p.setY(y)
    p.setTheta(theta)
    return p


# --- Measurement Model (measurement_model) ---
def measurement_model(distance, angle, p, landmark):
    """Computes likelihood p(z_t | x_t, m) using Gaussian on distance + angle error."""
    sigma_d = 15.0  # cm standard deviation for distance
    sigma_a = math.radians(10)  # radians for angle

    lx, ly = landmark
    dx, dy = lx - p.getX(), ly - p.getY()

    predicted_dist = math.sqrt(dx**2 + dy**2)
    predicted_angle = math.atan2(dy, dx) - p.getTheta()
    predicted_angle = (predicted_angle + math.pi) % (2 * math.pi) - math.pi

    dist_error = distance - predicted_dist
    angle_error = (angle - predicted_angle + math.pi) % (2 * math.pi) - math.pi

    weight_d = math.exp(-0.5 * (dist_error / sigma_d) ** 2)
    weight_a = math.exp(-0.5 * (angle_error / sigma_a) ** 2)
    return weight_d * weight_a


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

        # Use motor controls to update particles
        # XXX: Make the robot drive
        # XXX: You do this
        # --- MOTION UPDATE (simplified velocity + angular velocity model) ---
        dt = 0.1  # seconds per update (fixed small timestep)

        for p in particles:
            # Move particle according to velocity and angular velocity
            x = p.getX() + velocity * dt * math.cos(p.getTheta())
            y = p.getY() + velocity * dt * math.sin(p.getTheta())
            theta = (p.getTheta() + angular_velocity * dt) % (2 * math.pi)

            p.setX(x)
            p.setY(y)
            p.setTheta(theta)

        # Add Gaussian noise to simulate uncertainty
        particle.add_uncertainty(particles, sigma=6.0, sigma_theta=math.radians(2))
        # Fetch next frame
        colour = cam.get_next_frame()

        # Detect objects
        objectIDs, dists, angles = cam.detect_aruco_objects(colour)
        if not isinstance(objectIDs, type(None)):
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
                # XXX: Do something for each detected object - remember, the same ID may appear several times

            # --- MEASUREMENT UPDATE ---
            for p in particles:
                weight = 1.0
                for i in range(len(objectIDs)):
                    landmark_id = objectIDs[i]
                    if landmark_id not in landmarks:
                        continue
                    dist = dists[i]
                    ang = angles[i]
                    weight *= measurement_model(dist, ang, p, landmarks[landmark_id])
                p.setWeight(weight)

            # --- NORMALIZE WEIGHTS ---
            S = sum(p.getWeight() for p in particles)
            if S > 0:
                for p in particles:
                    p.setWeight(p.getWeight() / S)
            else:
                for p in particles:
                    p.setWeight(1.0 / len(particles))

            # --- RESAMPLING ---
            weights = [p.getWeight() for p in particles]
            indices = np.random.choice(len(particles), size=len(particles), p=weights)
            particles = [copy.deepcopy(particles[i]) for i in indices]

            # Optional: add 5% random new particles to maintain diversity
            for i in range(int(0.05 * len(particles))):
                particles[i] = particle.Particle(
                    600.0 * np.random.ranf() - 100.0,
                    600.0 * np.random.ranf() - 250.0,
                    np.mod(2.0 * np.pi * np.random.ranf(), 2.0 * np.pi),
                    1.0 / len(particles),
                )

            # Draw detected objects
            cam.draw_aruco_objects(colour)
        else:
            # No observation - reset weights to uniform distribution
            for p in particles:
                p.setWeight(1.0 / num_particles)

        est_pose = particle.estimate_pose(
            particles
        )  # The estimate of the robots current pose

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
