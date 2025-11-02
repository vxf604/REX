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


showGUI = True
onRobot = True
printer = print_path.PathPrinter(landmark_radius=20)

if onRobot:
    import robot


def isRunningOnArlo():
    return onRobot


# Keep robot and particles in sync for rotations
# If your robot API turns left for negative values, keep -1.0
# If it turns left for positive values, set to +1.0
ROTATE_CMD_SIGN = -1.0

CRED = (0, 0, 255)
CGREEN = (0, 255, 0)
CBLUE = (255, 0, 0)
CCYAN = (255, 255, 0)
CYELLOW = (0, 255, 255)
CMAGENTA = (255, 0, 255)
CWHITE = (255, 255, 255)
CBLACK = (0, 0, 0)

L1 = Landmark(x=0.0, y=0.0, color=CRED, ID=9, borderWidth_x=30, borderWidth_y=30)
L2 = Landmark(x=0.0, y=300.0, color=CGREEN, ID=7, borderWidth_x=30, borderWidth_y=-30)
L3 = Landmark(x=400.0, y=0.0, color=CYELLOW, ID=3, borderWidth_x=-30, borderWidth_y=30)
L4 = Landmark(x=400.0, y=300.0, color=CBLUE, ID=6, borderWidth_x=-30, borderWidth_y=-30)

landmarks = [L1, L2, L3, L4]
landmarkIDs = {l.ID: l for l in landmarks}
targets = [L2, L3, L4, L1]

obstacles_list = []


def jet(x):
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
    offsetX = 100
    offsetY = 50
    ymax = world.shape[0]
    world[:] = CWHITE
    max_weight = 0
    for particle in particles:
        max_weight = max(max_weight, particle.getWeight())
    for particle in particles:
        x = int(particle.getX() + offsetX)
        y = ymax - (int(particle.getY() + offsetY))
        colour = jet(particle.getWeight() / max_weight if max_weight > 0 else 0.0)
        cv2.circle(world, (x, y), 2, colour, 2)
        b = (
            int(particle.getX() + 15.0 * np.cos(particle.getTheta())) + offsetX,
            ymax
            - (int(particle.getY() + 15.0 * np.sin(particle.getTheta())) + offsetY),
        )
        cv2.line(world, (x, y), b, colour, 2)
    for landmark in landmarks:
        lm = (int(landmark.x + offsetX), int(ymax - (landmark.y + offsetY)))
        cv2.circle(world, lm, 5, landmark.color, 2)
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
        rotation(p, rot1, 0.10)  # gentler noise
        p.setX(p.getX() + transerror(0.3))
        p.setY(p.getY() + transerror(0.3))
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
        apply_sample_motion_model(particles, math.radians(-val), 0)
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
    e_l = np.array([lx - x, ly - y]) / d_i if d_i > 0 else np.array([0.0, 0.0])
    e_theta = np.array([math.cos(theta), math.sin(theta)])
    e_theta_hat = np.array([-math.sin(theta), math.cos(theta)])
    dot = float(np.clip(np.dot(e_l, e_theta), -1.0, 1.0))
    fi = sign(np.dot(e_l, e_theta_hat)) * math.acos(dot)
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
    x0, y0 = est_pose.getX(), est_pose.getY()
    theta = est_pose.getTheta()
    rx = dist * math.sin(angle)  # x left
    ry = dist * math.cos(angle)  # y forward
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
    n = np.linalg.norm(t)
    if n == 0:
        return 0.0
    t = t / n
    v = np.array([math.cos(robot_theta), math.sin(robot_theta)])
    dot = float(np.clip(np.dot(t, v), -1.0, 1.0))
    cross = v[0] * t[1] - v[1] * t[0]
    fi = math.acos(dot) * sign(cross)
    return math.degrees(fi)


def distance_to_target(est_pose, target):
    target_x, target_y = target[0], target[1]
    dx = target_x - est_pose.getX()
    dy = target_y - est_pose.getY()
    return math.sqrt(dx**2 + dy**2)


def execute_cmd(arlo, cmd):
    if not cmd:
        return
    movement, val = cmd
    if movement == "rotate":
        arlo.rotate_robot(val)
        time.sleep(0.5)
    elif movement == "forward":
        arlo.drive_forward_meter(val / 100.0)
        print(f" Driving forward {val} cm")
        time.sleep(0.5)
    elif movement == "stop":
        arlo.stop()


def distance(p1, p2):
    return float(np.linalg.norm(np.array(p1) - np.array(p2)))


def in_collision(point, obstacles, robot_radius=18):  # more conservative
    obstacle_radius = 22
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


def front_clear(arlo, dist_ok_cm=40):
    fl = ff = fr = None
    try:
        fl = arlo.read_left_ping()
    except Exception:
        pass
    try:
        ff = arlo.read_front_ping()
    except Exception:
        pass
    try:
        fr = arlo.read_right_ping()
    except Exception:
        pass
    vals = [v for v in [ff, fl, fr] if v is not None]
    if not vals:
        return True, 9999, fl, ff, fr
    m = min(vals)
    return (m >= dist_ok_cm), m, fl, ff, fr


def nearest_path_index(est_pose, path, start_idx):
    if not path or start_idx >= len(path):
        return start_idx
    px, py = est_pose.getX(), est_pose.getY()
    best_i, best_d = start_idx, float("inf")
    for i in range(start_idx, len(path)):
        qx, qy = path[i]
        dd = (qx - px) ** 2 + (qy - py) ** 2
        if dd < best_d:
            best_d, best_i = dd, i
    return best_i


def small_rotate_step(fi_deg, max_step=20.0):
    return max(-max_step, min(max_step, fi_deg))


def motor_control(
    state, est_pose, targets, seen2Landmarks, seen4Landmarks, obstacle_list, arlo
):
    if not hasattr(motor_control, "_search_rot"):
        motor_control._search_rot = 0.0
    if not hasattr(motor_control, "path"):
        motor_control.path = None
        motor_control.G = None
        motor_control.next_index = 1
    if not hasattr(motor_control, "_avoid_hits"):
        motor_control._avoid_hits = 0
    if not hasattr(motor_control, "_last_replan"):
        motor_control._last_replan = 0.0
    if not hasattr(motor_control, "_avoid_side"):
        motor_control._avoid_side = 1
    if not hasattr(motor_control, "_avoid_until"):
        motor_control._avoid_until = 0.0

    target = targets[0]
    target_pos = (target.x + target.borderWidth_x, target.y + target.borderWidth_y)

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

    fi = angle_to_target(est_pose, target_pos)
    d = distance_to_target(est_pose, target_pos)
    bearing = math.degrees(
        math.atan2(target_pos[1] - est_pose.getY(), target_pos[0] - est_pose.getX())
    )
    heading = math.degrees(est_pose.getTheta())
    fi = angle_to_target(est_pose, target_pos)
    print(f"bearing={bearing:.1f}°, heading={heading:.1f}°, fi={fi:.1f}°")
    align_ok = 4

    if state == "rotating":
        next_state = "forward" if abs(fi) < align_ok else "rotating"
        return ("rotate", fi), next_state

    if state == "forward":
        if d < 40:
            return ("rotate", fi), "finish_driving"
        if abs(fi) > align_ok:
            return ("rotate", fi), "forward"
        return ("forward", min(d, 20.0)), "forward"  # shorter chunks

    if state == "calculate_path":
        return (None, None), "follow_path"

    if state == "follow_path":
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
        G = motor_control.G

        if not path or len(path) < 2:
            return ("rotate", 20.0), "follow_path"

        printer.show_path_image(landmarks, obstacle_list, est_pose, target, G, path)

        waypoint = path[motor_control.next_index]
        fi = angle_to_target(est_pose, waypoint)
        d = distance_to_target(est_pose, waypoint)

        clear, mind, fl, ff, fr = front_clear(arlo, 40)
        if not clear:
            motor_control._avoid_until = time.time() + 3.0
            # turn away from the tighter side
            if fl is not None and fr is not None:
                motor_control._avoid_side = -1 if fl < fr else 1
            else:
                motor_control._avoid_side = getattr(motor_control, "_avoid_side", 1)
            return ("rotate", 20.0 * motor_control._avoid_side), "avoid"

        if d < 5.0:
            motor_control.next_index += 1
            if motor_control.next_index >= len(path):
                return (None, 0), "reached_target"
            return (None, 0), "follow_path"

        if abs(fi) > 4.0:
            return ("rotate", small_rotate_step(fi)), "follow_path"

        step = min(20.0, d)  # shorter chunks for safety
        return ("forward", step), "follow_path"

    if state == "avoid":
        clear, mind, fl, ff, fr = front_clear(arlo, 40)
        if clear:
            motor_control._avoid_hits = getattr(motor_control, "_avoid_hits", 0) + 1
            return ("forward", 15.0), "rejoin"
        if time.time() > getattr(motor_control, "_avoid_until", 0):
            if fl is not None and fr is not None:
                motor_control._avoid_side = -1 if fl < fr else 1
            else:
                motor_control._avoid_side = -getattr(motor_control, "_avoid_side", 1)
            motor_control._avoid_until = time.time() + 2.0
        return ("rotate", 12.0 * getattr(motor_control, "_avoid_side", 1)), "avoid"

    if state == "rejoin":
        path = motor_control.path
        if not path:
            return (None, 0), "follow_path"
        motor_control.next_index = nearest_path_index(
            est_pose, path, motor_control.next_index
        )
        if distance_to_target(
            est_pose, path[motor_control.next_index]
        ) < 8.0 and motor_control.next_index + 1 < len(path):
            motor_control.next_index += 1
        hits = getattr(motor_control, "_avoid_hits", 0)
        last = getattr(motor_control, "_last_replan", 0.0)
        if hits >= 2 and (time.time() - last) > 3.0:
            motor_control.path, motor_control.G = buildRRT(
                est_pose, obstacle_list, target
            )
            motor_control._avoid_hits = 0
            motor_control._last_replan = time.time()
            motor_control.next_index = 1
        return (None, 0), "follow_path"

    if state == "finish_driving":
        return ("forward", d), "reached_target"

    if state == "reached_target":
        if len(targets) > 0:
            targets.pop(0)
            motor_control.path = None
            motor_control.next_index = 1
            obstacles_list.clear()
            # force fresh perception before new plan
            return (None, 0), "fullSearch"
        return ("stop", None), "reached_target"


try:
    if showGUI:
        if not onRobot:
            WIN_RF1 = "Robot view"
            cv2.namedWindow(WIN_RF1, cv2.WINDOW_NORMAL)
        WIN_World = "World view"
        cv2.namedWindow(WIN_World, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_World, 520, 520)
        cv2.moveWindow(WIN_World, 720, 50)

    num_particles = 3000
    particles = initialize_particles(num_particles)
    state = "fullSearch"
    est_pose = particle_class.estimate_pose(particles)

    if isRunningOnArlo():
        arlo = robot.Robot()

    world = np.zeros((700, 700, 3), dtype=np.uint8)
    draw_world(est_pose, particles, world)
    cv2.imshow(WIN_World, world)

    print("Opening and initializing camera")
    if isRunningOnArlo():
        cam = camera.Camera(0, robottype="arlo", useCaptureThread=False)
    else:
        cam = camera.Camera(0, robottype="macbookpro", useCaptureThread=False)

    landmarks_seen = set()
    targetReached = True
    seeing = False
    seen2Landmarks = False

    while True:
        colour = cam.get_next_frame()
        print("state: ", state)
        objectIDs, dists, angles = cam.detect_aruco_objects(colour)
        if not isinstance(objectIDs, type(None)):
            obstacles_list = get_unique_obstacles(
                obstacles_list, objectIDs, dists, angles, landmarkIDs
            )
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
            weightSum = 0.0
            for particle in particles:
                likelyhood = 1.0
                for i in range(len(objectIDs)):
                    likelyhood *= measurement_model(
                        dists[i], angles[i], particle, landmarkIDs[objectIDs[i]]
                    )
                particle.setWeight(likelyhood)
                weightSum += likelyhood
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
            cam.draw_aruco_objects(colour)
            seeing = len(objectIDs) >= 1
        else:
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
            draw_world(est_pose, particles, world)
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
    cv2.destroyAllWindows()
    cam.terminateCaptureThread()
