import cv2
import cv2.aruco as aruco
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
import robot
import camera

SCALE = 100
arlo = robot.Robot()
landmarkChecker = landmark_checker.LandmarkChecker(landmark_radius=180, scale=SCALE)

onRobot = True  # Whether or not we are running on the Arlo robot
showGUI = False  # Whether or not to open GUI windows


def isRunningOnArlo():
    """Return True if we are running on Arlo, otherwise False.
    You can use this flag to switch the code from running on you laptop to Arlo - you need to do the programming here!
    """
    return onRobot


try:
    import robot

    onRobot = True
except ImportError:
    print("selflocalize.py: robot module not present - forcing not running on Arlo!")
    onRobot = False


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

landmarkIDs = [7, 2]
landmarks = {
    7: (0.0, 0.0),  # Coordinates for landmark 1
    2: (300.0, 0.0),  # Coordinates for landmark 2
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
def roterror(std_rot=math.radians(2.0)):
    return random.gauss(0.0, std_rot)


def transerror(trans1, std_trans=0.05):
    return random.gauss(0.0, abs(trans1) * std_trans)



def rotation1(p, rot1):
    theta = p.getTheta()
    theta = (theta + rot1 + roterror()) % (2 * np.pi)
    p.setTheta(theta)
    return p


def rotation2(p, rot2):
    theta = p.getTheta()
    theta = (theta + rot2 + roterror()) % (2 * np.pi)
    p.setTheta(theta)
    return p


def translation1(p, transl1):
    x = p.getX()
    y = p.getY()
    theta = p.getTheta()
    d = transl1 + transerror(transl1)
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


def normal_distribution(mu, sigma, x):
    return (1 / (math.sqrt(2 * math.pi) * sigma)) * math.exp(
        -0.5 * ((x - mu) / sigma) ** 2
    )
    
def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def predicted_distance(p, landmark):
    lx, ly = landmark
    x = p.getX()
    y = p.getY()
    return np.sqrt((lx - x) ** 2 + (ly - y) ** 2)

def predict_angle(p,landmark):
    lx, ly = landmark
    dx =lx -  p.getX()
    dy =ly - p.getY()
    angle = math.atan2(dy, dx)
    return normalize_angle(angle - p.getTheta())




def measurement_model(distance, angle, pt, landmark):
    predicted_dist = predicted_distance(pt, landmark)
    predict_ang = predict_angle(pt, landmark)
    
    angle_error = normalize_angle(angle - predict_ang)
    dist_error = distance - predicted_dist
    
    
    likelihood_dis = normal_distribution(0, 15.0, dist_error)

    likelihood_ang = normal_distribution(0, math.radians(25.0), angle_error)
    return likelihood_dis * likelihood_ang


# def MCL(
#     particles,
#     control_rtr,
#     detections,
#     LANDMARKS,
#     sig_d=10.0,
#     sig_b=math.radians(8.0),
#     angles_deg=True,
# ):
#     for particle in particles:
#         x = particle.getX()
#         y = particle.getY()
#         theta = particle.getTheta()

#         new_x, new_y, new_theta = sample_motion_model((x, y, theta))
#         weight = measurement_model((x, y, theta), LANDMARKS)
#         particle.setX(new_x)
#         particle.setY(new_y)
#         particle.setTheta(new_theta)
#         particle.setWeight(weight)


try:
    if showGUI:
        # Open windows
        WIN_RF1 = "Robot view"
        cv2.namedWindow(WIN_RF1)
        cv2.moveWindow(WIN_RF1, 50, 50)

        WIN_World = "World view"
        cv2.namedWindow(WIN_World)
        cv2.moveWindow(WIN_World, 500, 50)

    resample_count= 0
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
        resample_count += 1
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
        if resample_count == 20:
            landmarkcenter = ((landmarks[2][0] - landmarks[7][0])/2, landmarks[2][1] - landmarks[7][1])
            print("landmarkcenter:", landmarkcenter)
            distance = (((est_pose.getY() - landmarkcenter[1])**2 + ((est_pose.getX() - landmarkcenter[0])**2))**0.5)
            print("est_pose:", (est_pose.getX(), est_pose.getY(), est_pose.getTheta()))
            print("Distance to landmark center:", distance)
            
            angle_diff = math.atan(est_pose.getTheta() - math.atan2(landmarkcenter[1], landmarkcenter[0]))
            print("Angle difference:", angle_diff)
            print(arlo.rotate_robot(angle_diff))
            # print(arlo.drive_forward_meter(distance/100.0, 63,60))

        # Fetch next frame
        colour = cam.get_next_frame()

        # Detect objects
        objectIDs, dists, angles = cam.detect_aruco_objects(colour)
        if objectIDs is not None and len(objectIDs) > 0:
            angles = [-a for a in angles] 

            unique = {}
            for i, ID in enumerate(objectIDs):
                if ID not in unique:
                    unique[ID] = (dists[i], angles[i])
            objectIDs = list(unique.keys())
            dists = [unique[ID][0] for ID in objectIDs]
            angles = [unique[ID][1] for ID in objectIDs]
            
            
            for i in range(len(objectIDs)):
                print(f"Object ID = {objectIDs[i]}, Distance = {dists[i]}, angle = {angles[i]}")

            # --- Compute particle weights (combine all detections) ---
            
            for p in particles:
                p.setX(p.getX() + random.gauss(0, 5))  # jitter in mm/cm depending on your scale
                p.setY(p.getY() + random.gauss(0, 5))
                p.setTheta(p.getTheta() + random.gauss(0, math.radians(2)))
                
            for p in particles:
                weight = 1.0
                for i in range(len(objectIDs)):
                    landmark_id = int(objectIDs[i])
                    if landmark_id not in landmarks:
                        continue
                    dist_measured = dists[i]
                    ang_measured = angles[i]
                    weight *= measurement_model(dist_measured, ang_measured, p, landmarks[landmark_id])
                p.setWeight(weight)

            # --- Normalize weights ---
            S = sum(p.getWeight() for p in particles)
            if S <= 1e-300:
                for p in particles:
                    p.setWeight(1.0 / num_particles)
            else:
                invS = 1.0 / S
                for p in particles:
                    p.setWeight(p.getWeight() * invS)

            # --- Resample once ---
            weights = [p.getWeight() for p in particles]
            chosen = random.choices(particles, weights=weights, k=num_particles)

            # --- Replace worst 10% with random new particles ---
            frac_new = 0.1
            k_new = int(frac_new * num_particles)
            fresh = initialize_particles(k_new)
            particles = [copy.copy(p) for p in chosen[:-k_new]] + fresh
            
            for p in particles:
                p.setWeight(1.0 / num_particles)
                
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
