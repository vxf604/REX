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
import print_path


class Landmark:
    def __init__(self, x, y, color, ID, borderWidth_x, borderWidth_y):
        self.borderWidth_x = borderWidth_x
        self.borderWidth_y = borderWidth_y
        self.x = x
        self.y = y
        self.ID = ID
        self.color = color


# Flags
showGUI = True  # Whether or not to open GUI windows
onRobot = True  # Whether or not we are running on the Arlo robot
printer = print_path.PathPrinter(landmark_radius=20)  # mm


if onRobot:
    import robot


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
L1 = Landmark(x=0.0, y=0.0, color=CRED, ID=1, borderWidth_x=30, borderWidth_y=30)
L2 = Landmark(x=0.0, y=300.0, color=CGREEN, ID=2, borderWidth_x=30, borderWidth_y=-30)
L3 = Landmark(x=400.0, y=0.0, color=CYELLOW, ID=3, borderWidth_x=-30, borderWidth_y=30)
L4 = Landmark(x=400.0, y=300.0, color=CBLUE, ID=4, borderWidth_x=-30, borderWidth_y=-30)

landmarks = [L1, L2, L3, L4]

landmarkIDs = {l.ID: l for l in landmarks}

targets = [L2, L3, L4, L1]

obstacles_list = []


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


def calcutePos(est_pose, dist, angle):
    # est_pose in cm; angle in radians (check this!)
    x0, y0 = est_pose.getX(), est_pose.getY()
    theta = est_pose.getTheta()  # robot heading in radians

    # ray in robot frame
    rx = dist * math.cos(angle)
    ry = dist * math.sin(angle)

    # rotate robot->world by theta
    c, s = math.cos(theta), math.sin(theta)
    wx = x0 + c * rx - s * ry
    wy = y0 + s * rx + c * ry
    return wx, wy


def get_unique_obstacles(obstacles_list, objectIDs, dists, angles, landmarkIDs):
    uniqueIDs = set(objectIDs)
    obstaclesListIDs = [o.ID for o in obstacles_list]

    for uid in uniqueIDs:
        indices = [i for i, id in enumerate(objectIDs) if id == uid]
        closest_id = min(indices, key=lambda i: dists[i])
        if uid not in landmarkIDs and uid not in obstaclesListIDs:
            id = objectIDs[closest_id]
            angle = angles[closest_id]
            dist = dists[closest_id]
            print(
                f"obstacle id: {id}, dist: {dist}, est pose: {est_pose.getX()}, {est_pose.getY()}"
            )
            x, y = calcutePos(est_pose, dist, angle)
            obstacle = Landmark(x, y, CBLACK, id, 10, 10)
            obstacles_list.append(obstacle)
            obstaclesListIDs.append(id)
    return obstacles_list


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


def distance(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def in_collision(point, obstacles, robot_radius=15):
    obstacle_radius = 18
    x, y = point
    for obstacle in obstacles:
        map_x, map_y = obstacle.x, obstacle.y
        obstacle_pos = (map_x, map_y)
        d = distance((x, y), obstacle_pos)
        if d <= (obstacle_radius + robot_radius):
            return True
    return False


def randConf():
    return (random.uniform(0, 400), random.uniform(0, 300))


def NEAREST_VERTEX(v, G):
    min = distance(G[0], v)
    minV = G[0]
    for i in range(len(G)):
        d = distance(v, G[i])
        if d < min or i == 0:
            min = d
            minV = G[i]
    return minV


def Steer(q_near, q_rand, delta_q=40):
    dx, dy = q_rand[0] - q_near[0], q_rand[1] - q_near[1]
    d = distance(q_near, q_rand)

    if d < delta_q:
        return q_rand
    else:
        return (q_near[0] + delta_q * dx / d, q_near[1] + delta_q * dy / d)


def buildRRT(est_pose, obstacles_list, goal, delta_q=40):

    start = (est_pose.getX(), est_pose.getY())
    G = [start]
    parent = {0: None}
    goal_pos = (goal.x + goal.borderWidth_x, goal.y + goal.borderWidth_y)
    goal_index = None
    i = 0

    while goal_index is None:
        i += 1
        q_rand = randConf()
        q_near = NEAREST_VERTEX(q_rand, G)
        q_new = Steer(q_near, q_rand, delta_q)

        if in_collision(q_new, obstacles_list):
            continue

        G.append(q_new)
        parent[len(G) - 1] = G.index(q_near)

        if distance(q_new, goal_pos) < delta_q:
            goal_index = len(G) - 1

    path = []
    node = goal_index
    while node is not None:
        path.append(G[node])
        node = parent[node]
    path.reverse()

    path.append(goal_pos)

    return path, G


def avoidance(arlo, est_pose, obstacles_list):
    if not obstacles_list:
        return False

    robot_x, robot_y = est_pose.getX(), est_pose.getY()

    for close_obstacle in obstacles_list:
        distance = math.sqrt(
            (close_obstacle.y - robot_y) ** 2 + (close_obstacle.x - robot_x) ** 2
        )

        if distance < 40:
            left = arlo.read_left_ping_sensor()
            right = arlo.read_right_ping_sensor()
            front = arlo.read_front_ping_sensor()

            if left < 400 or right < 400 or front < 400:  ## mm
                if right > left:
                    return "right"
                else:
                    return "left"

    return None


def read_front_cm(arlo):
    try:
        return arlo.read_front_ping_sensor() / 10.0
    except Exception:
        return 9999.0


def read_left_cm(arlo):
    try:
        return arlo.read_left_ping_sensor() / 10.0
    except Exception:
        return 9999.0


def read_right_cm(arlo):
    try:
        return arlo.read_right_ping_sensor() / 10.0
    except Exception:
        return 9999.0


def motor_control(
    state, est_pose, targets, seen2Landmarks, seen4Landmarks, obstacle_list, arlo
):
    # interne tilstande
    if not hasattr(motor_control, "_search_rot"):
        motor_control._search_rot = 0.0
    if not hasattr(motor_control, "path"):
        motor_control.path = None
        motor_control.G = None
        motor_control.next_index = 1

    # ingen targets tilbage
    if not targets:
        return ("stop", None), "reached_target"

    target = targets[0]
    target_pos = (target.x + target.borderWidth_x, target.y + target.borderWidth_y)

    # simple debug
    print(
        f"est_pose: x: {est_pose.getX():.3f}, y: {est_pose.getY():.3f}, theta: {math.degrees(est_pose.getTheta()):.1f}"
    )

    # high level states for at komme i gang
    if state == "searching":
        if seen2Landmarks:
            return (None, 0), "follow_path"
        return ("rotate", 20.0), "searching"

    if state == "fullSearch":
        motor_control._search_rot += 20.0
        if seen4Landmarks or motor_control._search_rot >= 360.0:
            motor_control._search_rot = 0.0
            return (None, 0), "follow_path"
        else:
            return ("rotate", 20.0), "fullSearch"

    if state == "calculate_path":
        # vi beregner ruten i follow_path
        return (None, 0), "follow_path"

    if state == "rotating":
        # denne bruges ikke aktivt i den simple kontrol, men beholdes for kompatibilitet
        return (None, 0), "follow_path"

    if state == "forward":
        # denne bruges heller ikke aktivt i den simple kontrol
        return (None, 0), "follow_path"

    if state == "follow_path":
        # konstanter
        ALIGN_DEG = 10.0  # større giver færre ryk
        MAX_TURN = 30.0  # maks pr rotate kommando
        STEP_CM = 30.0  # frem i små hop
        GOAL_TOL = 12.0  # hvornår sidste waypoint er nået
        MARGIN = 5.0  # sikkerhed på frontafstand
        SIDE_ENTER = 12.0  # sideafstand der udløser nudge
        NUDGE_DEG = 10.0  # lille nudge vinkel

        # plan
        need_plan = motor_control.path is None or motor_control.next_index >= (
            len(motor_control.path) if motor_control.path else 0
        )
        if need_plan:
            motor_control.path, motor_control.G = buildRRT(
                est_pose, obstacle_list, target
            )
            motor_control.next_index = 1
            print("Path:", motor_control.path)

        path = motor_control.path
        plen = len(path) if path else 0
        print(f"[path] idx={motor_control.next_index} len={plen}")

        if not path or len(path) < 2:
            return ("rotate", 20.0), "follow_path"

        # aktuel waypoint
        i = motor_control.next_index
        wp = path[i]
        fi = angle_to_target(est_pose, wp)
        d = distance_to_target(est_pose, wp)
        last = i == len(path) - 1

        # er vi i mål ved sidste punkt
        if last and d <= GOAL_TOL:
            return (None, 0), "reached_target"

        # 1) Align altid først, så vi undgår ryk
        if abs(fi) > ALIGN_DEG:
            turn = max(-MAX_TURN, min(MAX_TURN, fi))
            return ("rotate", turn), "follow_path"

        # 2) Sensor check
        front = read_front_cm(arlo)
        left = read_left_cm(arlo)
        right = read_right_cm(arlo)

        block_front = min(d, STEP_CM) + MARGIN

        # hvis der er noget foran, drej en smule væk fra den nærmeste side
        if front < block_front:
            if left < right:
                return ("rotate", NUDGE_DEG), "follow_path"  # drej mod højre
            elif right < left:
                return ("rotate", -NUDGE_DEG), "follow_path"  # drej mod venstre
            else:
                return ("rotate", NUDGE_DEG), "follow_path"

        # hvis en side er tæt, lav en lille nudge væk fra den
        if left < SIDE_ENTER and right >= SIDE_ENTER:
            return ("rotate", NUDGE_DEG), "follow_path"
        if right < SIDE_ENTER and left >= SIDE_ENTER:
            return ("rotate", -NUDGE_DEG), "follow_path"
        if left < SIDE_ENTER and right < SIDE_ENTER:
            return ("rotate", NUDGE_DEG), "follow_path"

        # 3) Waypoint hop hvis vi allerede er lige ved det
        if d < 5.0 and not last:
            motor_control.next_index = i + 1
            return (None, 0), "follow_path"

        # 4) Kør frem i et step
        step = min(STEP_CM, d)

        # hvis næste hop vil ramme waypointet, bump index nu, men kør stadig step
        if d <= STEP_CM + 2.0 and not last:
            motor_control.next_index = i + 1

        # hvis vi er på sidste punkt og det er inden for step, så afslut efter kørslen
        if last and d <= STEP_CM:
            return ("forward", d), "reached_target"

        return ("forward", step), "follow_path"

    if state == "avoidance":
        # ikke brugt i denne simple udgave
        return (None, 0), "follow_path"

    if state == "finish_driving":
        # ikke brugt i denne simple udgave
        return (None, 0), "follow_path"

    if state == "reached_target":
        if len(targets) > 0:
            targets.pop(0)
            # reset plan og index for næste target
            motor_control.path = None
            motor_control.G = None
            motor_control.next_index = 1
            return ("rotate", 20.0), "fullSearch"
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

    # Initialize particles
    num_particles = 3000
    particles = initialize_particles(num_particles)
    state = "fullSearch"
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
            obstacles_list = get_unique_obstacles(
                obstacles_list, objectIDs, dists, angles, landmarkIDs
            )
            for obstacle in obstacles_list:

                print(f"Obstacle: x: {obstacle.x}, y: {obstacle.y}, ID: {obstacle.ID}")
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

        seen2Landmarks = len(landmarks_seen) >= 2
        seen4Landmarks = len(landmarks_seen) >= 4
        if onRobot:
            cmd, state = motor_control(
                state,
                est_pose,
                targets,
                seen2Landmarks,
                seen4Landmarks,
                obstacles_list,
                arlo,
            )
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
