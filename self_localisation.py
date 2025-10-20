import cv2
import cv2.aruco as aruco
from pprint import *
from time import sleep
import numpy as np
import random
import camera
import particle
import math
import copy

onRobot = True  # Whether or not we are running on the Arlo robot
showGUI = False  # Whether or not to open GUI windows



def isRunningOnArlo():
    """Return True if we are running on Arlo, otherwise False."""
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
CYELLOW = (0, 255, 255)
CMAGENTA = (255, 0, 255)
CWHITE = (255, 255, 255)


landmarkIDs = [7, 2]
landmarks = {
    7: (0.0, 0.0), #Coordinates of landmark 7
    2: (200.0, 0.0), #Coordinates of landmark 2
}
landmark_colors = [CRED, CGREEN] # Colors for drawing landmarks


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
    
    world[:] = CWHITE # Clear the background to white
    
    #Find the largest weight
    max_weight = 0
    for particle in particles:
        max_weight = max(max_weight, particle.getWeight())

    
    # Draw particles
    for p in particles:
        x = int(p.getX() + offsetX)
        y = ymax - (int(p.getY() + offsetY))
        colour = jet(p.getWeight() / max_weight) if max_weight > 0 else jet(0)
        cv2.circle(world, (x, y), 2, colour, 2)
        b = (
            int(p.getX() + 15.0 * np.cos(p.getTheta())) + offsetX,
            ymax - (int(p.getY() + 15.0 * np.sin(p.getTheta())) + offsetY),
        )
        cv2.line(world, (x, y), b, colour, 2)

    # Draw landmarks
    for i, ID in enumerate(landmarkIDs):
        lm = (int(landmarks[ID][0] + offsetX), int(ymax - (landmarks[ID][1] + offsetY)))
        cv2.circle(world, lm, 5, landmark_colors[i], 2)

    a = (int(est_pose.getX()) + offsetX, ymax - (int(est_pose.getY()) + offsetY))
    b = (
        int(est_pose.getX() + 15.0 * np.cos(est_pose.getTheta())) + offsetX,
        ymax - (int(est_pose.getY() + 15.0 * np.sin(est_pose.getTheta())) + offsetY),
    )
    cv2.circle(world, a, 5, CMAGENTA, 2)
    cv2.line(world, a, b, CMAGENTA, 2)


# Particle initialization
def initialize_particles(num_particles):
    particles = []
    for i in range(num_particles):
        p = particle.Particle(
            600.0 * np.random.ranf() - 100.0,
            600.0 * np.random.ranf() - 250.0,
            np.mod(2.0 * np.pi * np.random.ranf(), 2.0 * np.pi),
            1.0 / num_particles,
        )
        particles.append(p)
    return particles


def roterror(std_rot=math.radians(2.0)):
    return random.gauss(0.0, std_rot)


def transerror(trans1, std_trans=0.05):
    return random.gauss(0.0, abs(trans1) * std_trans)


def rotation1(p, rot1):
    theta = (p.getTheta() + rot1 + roterror()) % (2 * np.pi)
    p.setTheta(theta)
    return p


def rotation2(p, rot2):
    theta = (p.getTheta() + rot2 + roterror()) % (2 * np.pi)
    p.setTheta(theta)
    return p


def translation1(p, transl1):
    d = transl1 + transerror(transl1)
    x = p.getX() + d * np.cos(p.getTheta())
    y = p.getY() + d * np.sin(p.getTheta())
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


def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def predicted_distance(p, landmark):
    lx, ly = landmark
    return np.sqrt((lx - p.getX()) ** 2 + (ly - p.getY()) ** 2)


def predict_angle(p, landmark):
    lx, ly = landmark
    dx, dy = lx - p.getX(), ly - p.getY()
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


# --- Main ---
try:
    if showGUI:
        WIN_RF1 = "Robot view"
        cv2.namedWindow(WIN_RF1)
        cv2.moveWindow(WIN_RF1, 50, 50)
        WIN_World = "World view"
        cv2.namedWindow(WIN_World)
        cv2.moveWindow(WIN_World, 500, 50)
    else:
        print("GUI disabled")

    num_particles = 1000
    particles = initialize_particles(num_particles)
    est_pose = particle.estimate_pose(particles)

    if isRunningOnArlo():
        arlo = robot.Robot()

    world = np.zeros((500, 500, 3), dtype=np.uint8)
    draw_world(est_pose, particles, world)

    print("Opening and initializing camera")
    cam = camera.Camera(0, robottype="arlo" if onRobot else "macbookpro", useCaptureThread=False)

    while True:
        if cv2.waitKey(10) == ord("q"):
            break

        if onRobot:
            target = (
                (landmarks[2][0] + landmarks[7][0]) / 2,
                (landmarks[2][1] + landmarks[7][1]) / 2,
            )

            distance_cm = math.sqrt(
                (target[0] - est_pose.getX()) ** 2 + (target[1] - est_pose.getY()) ** 2
            )
            target_angle = math.atan2(target[1] - est_pose.getY(), target[0] - est_pose.getX())
            angle_diff = (target_angle - est_pose.getTheta() + math.pi) % (2 * math.pi) - math.pi

            print(f"Rotating {angle_diff:.2f} radians")
            arlo.rotate_robot(angle_diff)

            partial_distance_cm = distance_cm / 4.0
            partial_distance_m = partial_distance_cm / 100.0
            print(f"Driving first 1/4: {partial_distance_m:.2f} meters")
            arlo.drive_forward_meter(partial_distance_m, 67, 64)

            for i in range(len(particles)):
                particles[i] = sample_motion_model(particles[i], angle_diff, partial_distance_cm, 0.0)

            colour = cam.get_next_frame()
            objectIDs, dists, angles = cam.detect_aruco_objects(colour)

            if objectIDs is not None and len(set(objectIDs)) >= 2:
                print("Still see two landmarks, re-localizing...")
                for p in particles:
                    weight = 1.0
                    for i, ID in enumerate(objectIDs):
                        if ID not in landmarks:
                            continue
                        dist_measured = dists[i]
                        ang_measured = -angles[i]
                        weight *= measurement_model(dist_measured, ang_measured, p, landmarks[ID])
                    p.setWeight(weight)

                S = sum(p.getWeight() for p in particles)
                if S <= 1e-12:
                    for p in particles:
                        p.setWeight(1.0 / len(particles))
                else:
                    invS = 1.0 / S
                    for p in particles:
                        p.setWeight(p.getWeight() * invS)

                weights = [p.getWeight() for p in particles]
                particles = random.choices(particles, weights=weights, k=len(particles))
                est_pose = particle.estimate_pose(particles)

                new_distance_cm = math.sqrt(
                    (target[0] - est_pose.getX()) ** 2 + (target[1] - est_pose.getY()) ** 2
                )
                remaining_distance_m = new_distance_cm / 100.0
                print(f"Driving remaining {remaining_distance_m:.2f} meters")
                arlo.drive_forward_meter(remaining_distance_m, 67, 64)

                for i in range(len(particles)):
                    particles[i] = sample_motion_model(particles[i], 0.0, new_distance_cm, 0.0)
            else:
                print("Lost landmark tracking — driving remaining 3/4 without update.")
                remaining_distance_cm = distance_cm * (3.0 / 4.0)
                remaining_distance_m = remaining_distance_cm / 100.0
                arlo.drive_forward_meter(remaining_distance_m, 67, 64)
                for i in range(len(particles)):
                    particles[i] = sample_motion_model(particles[i], 0.0, remaining_distance_cm, 0.0)


        colour = cam.get_next_frame()
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

            for p in particles:
                weight = 1.0
                for i, ID in enumerate(objectIDs):
                    if ID not in landmarks:
                        continue
                    weight *= measurement_model(dists[i], angles[i], p, landmarks[ID])
                p.setWeight(weight)

            S = sum(p.getWeight() for p in particles)
            if S <= 1e-300:
                for p in particles:
                    p.setWeight(1.0 / num_particles)
            else:
                invS = 1.0 / S
                for p in particles:
                    p.setWeight(p.getWeight() * invS)

            weights = [p.getWeight() for p in particles]
            chosen = random.choices(particles, weights=weights, k=num_particles)

            frac_new = 0.1
            k_new = int(frac_new * num_particles)
            fresh = initialize_particles(k_new)
            particles = [copy.copy(p) for p in chosen[:-k_new]] + fresh

            for p in particles:
                p.setWeight(1.0 / num_particles)

            particle.add_uncertainty(particles, sigma=5.0, sigma_theta=math.radians(3.0))
            cam.draw_aruco_objects(colour)

        est_pose = particle.estimate_pose(particles)

        if showGUI:
            draw_world(est_pose, particles, world)
            cv2.imshow(WIN_RF1, colour)
            cv2.imshow(WIN_World, world)

finally:
    cv2.destroyAllWindows()
    cam.terminateCaptureThread()
