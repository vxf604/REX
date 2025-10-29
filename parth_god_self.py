import copy
import math
import os
import random
import time
import sys

import cv2
import numpy as np

import particle as particle_class
import camera

# Flags
showGUI = True
onRobot = True

try:
    import robot

    onRobot = True
    showGUI = False  # generally headless when on robot
except Exception:
    onRobot = False
    showGUI = True
    robot = None

# Color constants (BGR)
CRED = (0, 0, 255)
CGREEN = (0, 255, 0)
CBLUE = (255, 0, 0)
CMAGENTA = (255, 0, 255)
CWHITE = (255, 255, 255)

# Known landmarks (IDs -> (x_cm, y_cm))
landmarkIDs = [6, 2]
landmarks = {6: (0.0, 0.0), 2: (120.0, 0.0)}
landmark_colors = [CRED, CGREEN]

# Utility: color map
def jet(x):
    """Map [0..1] to BGR color (same interface as before)."""
    x = float(np.clip(x, 0.0, 1.0))
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
    return (int(255.0 * b), int(255.0 * g), int(255.0 * r))


def draw_world(est_pose, particles, world):
    """
    Draw particles, landmarks, and estimated pose.
    Note: we define world coordinates: +x to right, +y up.
    Screen y is downward, so we flip when drawing.
    """
    offsetX = 100
    offsetY = 100
    world[:] = CWHITE
    ymax = world.shape[0]

    # Find max weight for colouring (avoid zero divide)
    weights = [p.getWeight() for p in particles]
    max_w = max(weights) if len(weights) > 0 else 1e-12

    for p in particles:
        xw = int(p.getX() + offsetX)
        yw = int(ymax - (p.getY() + offsetY))
        colour = jet(p.getWeight() / max_w if max_w > 0 else 0.0)
        cv2.circle(world, (xw, yw), 2, colour, -1)
        bx = int(p.getX() + 15.0 * math.cos(p.getTheta()) + offsetX)
        by = int(ymax - (p.getY() + 15.0 * math.sin(p.getTheta()) + offsetY))
        cv2.line(world, (xw, yw), (bx, by), colour, 1)

    # landmarks
    for i, ID in enumerate(landmarkIDs):
        lx, ly = landmarks[ID]
        lm = (int(lx + offsetX), int(ymax - (ly + offsetY)))
        cv2.circle(world, lm, 6, landmark_colors[i], -1)

    # estimated pose
    if est_pose is not None:
        ex = int(est_pose[0] + offsetX)
        ey = int(ymax - (est_pose[1] + offsetY))
        bx = int(est_pose[0] + 20.0 * math.cos(est_pose[2]) + offsetX)
        by = int(ymax - (est_pose[1] + 20.0 * math.sin(est_pose[2]) + offsetY))
        cv2.circle(world, (ex, ey), 6, CMAGENTA, -1)
        cv2.line(world, (ex, ey), (bx, by), CMAGENTA, 2)


# Initialize particles uniformly in area
def initialize_particles(num_particles, x_range=(-100, 500), y_range=(-250, 350)):
    particles = []
    for _ in range(num_particles):
        x = random.uniform(*x_range)
        y = random.uniform(*y_range)
        theta = random.uniform(-math.pi, math.pi)
        p = particle_class.Particle(x, y, theta, 1.0 / num_particles)
        particles.append(p)
    return particles


# Motion noise parameters (alphas) - standard velocity motion model style
ALPHA1 = 0.03  # rot noise proportional to rot^2
ALPHA2 = 0.02  # rot noise proportional to trans^2
ALPHA3 = 0.04  # trans noise proportional to trans^2
ALPHA4 = 0.02  # trans noise proportional to rot^2


def sample_motion(p, rot1, trans, rot2):
    """Apply probabilistic motion update to particle p using rot1, trans, rot2."""
    # Add noise to controls (Thrun-style)
    std_rot1 = math.sqrt(ALPHA1 * rot1 * rot1 + ALPHA2 * trans * trans)
    std_trans = math.sqrt(ALPHA3 * trans * trans + ALPHA4 * (rot1 * rot1 + rot2 * rot2))
    std_rot2 = math.sqrt(ALPHA1 * rot2 * rot2 + ALPHA2 * trans * trans)

    rot1_hat = rot1 + random.gauss(0.0, std_rot1)
    trans_hat = trans + random.gauss(0.0, std_trans)
    rot2_hat = rot2 + random.gauss(0.0, std_rot2)

    x = p.getX() + trans_hat * math.cos(p.getTheta() + rot1_hat)
    y = p.getY() + trans_hat * math.sin(p.getTheta() + rot1_hat)
    theta = p.getTheta() + rot1_hat + rot2_hat

    p.setX(x)
    p.setY(y)
    p.setTheta(((theta + math.pi) % (2.0 * math.pi)) - math.pi)


def apply_motion_update(particles, rot1, trans, rot2):
    for p in particles:
        sample_motion(p, rot1, trans, rot2)


# Measurement model using log probabilities for numeric stability
SIGMA_D = 15.0  # cm
SIGMA_A = 0.1  # rad


def measurement_log_prob(distance_obs, angle_obs, particle, landmark_pos):
    """
    Compute log probability of observing (distance_obs, angle_obs) given particle and landmark.
    """
    lx, ly = landmark_pos
    x, y, theta = particle.getX(), particle.getY(), particle.getTheta()

    # predicted measurement
    dx = lx - x
    dy = ly - y
    d_pred = math.hypot(dx, dy)
    # bearing relative to robot heading:
    bearing = math.atan2(dy, dx) - theta
    # normalize bearing to [-pi, pi]
    bearing = (bearing + math.pi) % (2.0 * math.pi) - math.pi

    # Gaussian log probabilities
    # log p(distance)
    log_p_d = -0.5 * ((distance_obs - d_pred) ** 2) / (SIGMA_D * SIGMA_D) - math.log(
        math.sqrt(2.0 * math.pi) * SIGMA_D
    )
    # log p(angle)
    # angle difference normalized
    da = (angle_obs - bearing + math.pi) % (2.0 * math.pi) - math.pi
    log_p_a = -0.5 * (da ** 2) / (SIGMA_A * SIGMA_A) - math.log(math.sqrt(2.0 * math.pi) * SIGMA_A)

    return log_p_d + log_p_a


def get_unique_landmarks(objectIDs, dists, angles, knownIDs):
    """Return lists of unique detected known landmarks (closest measurement per ID)."""
    if objectIDs is None or len(objectIDs) == 0:
        return [], [], []
    unique = {}
    for i, id_ in enumerate(objectIDs):
        if id_ not in knownIDs:
            continue
        if id_ not in unique or dists[i] < unique[id_][0]:
            unique[id_] = (dists[i], angles[i], i)
    detectedIDs = []
    detectedDists = []
    detectedAngles = []
    for k, v in unique.items():
        detectedIDs.append(k)
        detectedDists.append(v[0])
        detectedAngles.append(v[1])
    return detectedIDs, detectedDists, detectedAngles


def normalize_weights_from_log(log_weights):
    """Convert log-weights to normalized weights array (sum to 1) safely."""
    if len(log_weights) == 0:
        return np.array([])
    maxlog = max(log_weights)
    exps = [math.exp(lw - maxlog) for lw in log_weights]
    s = sum(exps)
    if s == 0:
        n = len(exps)
        return np.array([1.0 / n] * n)
    return np.array([e / s for e in exps])


def low_variance_resample(particles, weights):
    """Systematic (low-variance) resampling. Returns new particles (deep copies)."""
    N = len(particles)
    positions = (np.arange(N) + random.random()) / N
    cumulative = np.cumsum(weights)
    i, j = 0, 0
    new_particles = []
    while i < N:
        while positions[i] > cumulative[j]:
            j += 1
        new_particles.append(copy.copy(particles[j]))
        # reset weight for new particle (uniform after resample)
        new_particles[-1].setWeight(1.0 / N)
        i += 1
    return new_particles


def estimate_pose_from_particles(particles):
    """Return weighted mean pose (x, y, theta)."""
    weights = np.array([p.getWeight() for p in particles])
    s = weights.sum()
    if s == 0:
        # uniform fallback
        xs = np.array([p.getX() for p in particles])
        ys = np.array([p.getY() for p in particles])
        thetas = np.array([p.getTheta() for p in particles])
        return (float(xs.mean()), float(ys.mean()), float(np.arctan2(np.mean(np.sin(thetas)), np.mean(np.cos(thetas)))))
    weights /= s
    x = sum(p.getX() * w for p, w in zip(particles, weights))
    y = sum(p.getY() * w for p, w in zip(particles, weights))
    # circular mean for angle
    sin_sum = sum(math.sin(p.getTheta()) * w for p, w in zip(particles, weights))
    cos_sum = sum(math.cos(p.getTheta()) * w for p, w in zip(particles, weights))
    theta = math.atan2(sin_sum, cos_sum)
    return (x, y, theta)


def particle_spread(particles):
    xs = np.array([p.getX() for p in particles])
    ys = np.array([p.getY() for p in particles])
    return xs.std(), ys.std()


# --- Main ---
def main():
    global showGUI, onRobot

    # Single robot instance if available
    arlo = None
    if robot is not None and onRobot:
        try:
            arlo = robot.Robot()
            onRobot = True
            showGUI = False
        except Exception as e:
            print("Failed to initialize robot interface:", e)
            arlo = None
            onRobot = False
            showGUI = True

    # GUI setup
    if showGUI:
        WIN_RF1 = "Robot view"
        cv2.namedWindow(WIN_RF1)
        cv2.moveWindow(WIN_RF1, 50, 50)
        WIN_World = "World view"
        cv2.namedWindow(WIN_World)
        cv2.moveWindow(WIN_World, 500, 50)

    # parameters
    num_particles = 1000
    particles = initialize_particles(num_particles)
    # initial uniform weights are already set in Particle init

    est_pose = estimate_pose_from_particles(particles)

    # create camera
    cam = None
    try:
        if onRobot:
            cam = camera.Camera(0, robottype="arlo", useCaptureThread=False)
        else:
            cam = camera.Camera(0, robottype="macbookpro", useCaptureThread=False)
    except Exception as e:
        print("Camera init failed:", e)
        cam = None

    world = np.zeros((500, 500, 3), dtype=np.uint8)

    target = (60.0, 0.0)
    landmarks_seen = set()
    max_iterations = 1000
    iteration = 0

    # Safety: max loop time or iterations
    try:
        while iteration < max_iterations:
            iteration += 1

            # 1) Decide robot action using estimated pose (closed-loop)
            robot_x, robot_y, robot_theta = est_pose
            dx = target[0] - robot_x
            dy = target[1] - robot_y
            distance_to_target = math.hypot(dx, dy)

            # If close enough, stop
            if distance_to_target < 5.0:
                if arlo is not None and hasattr(arlo, "stop"):
                    arlo.stop()
                print("Reached target (est).")
                break

            # Compute desired heading
            desired_theta = math.atan2(dy, dx)
            angle_error = (desired_theta - robot_theta + math.pi) % (2.0 * math.pi) - math.pi

            # Plan small motion for this loop: rotate towards target then forward a controlled amount
            # Cap rotation and translation per iteration to stable values
            max_rot_deg = 20.0
            rot_deg = max(-max_rot_deg, min(max_rot_deg, math.degrees(angle_error)))
            rot_rad = math.radians(rot_deg)
            trans = min(20.0, distance_to_target)  # cm

            # Execute motion on robot if present (non-blocking API assumed). Use odometry if available.
            # We'll apply motion model using measured odometry if provided by robot; otherwise we simulate using commanded motion.
            odometry_available = False
            od_delta_rot1 = od_delta_trans = od_delta_rot2 = 0.0

            if arlo is not None:
                try:
                    # send rotate
                    if abs(rot_deg) > 1e-3:
                        arlo.rotate_robot(rot_deg)
                        time.sleep(0.5)  # allow rotation to occur (adapt to robot API)
                    # send forward motion
                    if trans > 1e-3:
                        arlo.drive_forward_meter(trans / 100.0)
                        time.sleep(max(0.5, trans / 100.0))  # allow drive to complete (tunable)
                    # try to read odometry if implemented
                    if hasattr(arlo, "get_odometry"):
                        odometry_available = True
                        od_delta_rot1, od_delta_trans, od_delta_rot2 = arlo.get_odometry()
                    else:
                        # no odometry method - assume commanded values as deltas
                        od_delta_rot1, od_delta_trans, od_delta_rot2 = rot_rad, trans, 0.0
                except Exception as e:
                    print("Warning: robot motion failed or is slow:", e)
                    # fallback to commanded
                    od_delta_rot1, od_delta_trans, od_delta_rot2 = rot_rad, trans, 0.0
            else:
                # no robot - simulate commands directly
                od_delta_rot1, od_delta_trans, od_delta_rot2 = rot_rad, trans, 0.0

            # 2) Motion update (use odometry deltas)
            apply_motion_update(particles, od_delta_rot1, od_delta_trans, od_delta_rot2)

            # 3) Get camera frame and detect landmarks
            objectIDs = dists = angles = None
            frame = None
            if cam is not None:
                try:
                    frame = cam.get_next_frame()
                    det = cam.detect_aruco_objects(frame)
                    if det is not None:
                        objectIDs, dists, angles = det
                except Exception as e:
                    print("Camera read/detect error:", e)
                    objectIDs = dists = angles = None

            # 4) Measurement update (if any known landmark observed)
            detectedIDs, detectedDists, detectedAngles = get_unique_landmarks(objectIDs, dists, angles, landmarkIDs)
            if len(detectedIDs) > 0:
                # compute log-likelihoods per particle
                log_weights = []
                for p in particles:
                    lw = 0.0
                    for i, lm_id in enumerate(detectedIDs):
                        lm_pos = landmarks[lm_id]
                        lw += measurement_log_prob(detectedDists[i], detectedAngles[i], p, lm_pos)
                    log_weights.append(lw)
                # convert log-weights to normalized weights
                new_weights = normalize_weights_from_log(log_weights)
                for i, p in enumerate(particles):
                    p.setWeight(float(new_weights[i]))
                # resample (systematic)
                particles = low_variance_resample(particles, new_weights)
            else:
                # No observation -> keep current weights but optionally add slight diffusion to avoid overconfidence
                # (do nothing here so previous belief persists)
                pass

            # 5) Estimate pose
            est_pose = estimate_pose_from_particles(particles)

            # 6) Visualization / save
            if showGUI:
                draw_world(est_pose, particles, world)
                if frame is not None:
                    if hasattr(cam, "draw_aruco_objects"):
                        try:
                            cam.draw_aruco_objects(frame)
                        except Exception:
                            pass
                    cv2.imshow(WIN_RF1, frame)
                cv2.imshow(WIN_World, world)
                key = cv2.waitKey(10) & 0xFF
                if key == ord("q"):
                    print("User requested quit.")
                    break
            else:
                # headless: save occasional snapshots, not every frame
                if iteration % 50 == 0:
                    draw_world(est_pose, particles, world)
                    os.makedirs("images", exist_ok=True)
                    cv2.imwrite(os.path.join("images", f"world_iter_{iteration}.png"), world)

            # 7) Termination check: particle spread small enough
            sx, sy = particle_spread(particles)
            if sx < 5.0 and sy < 5.0:
                print(f"Converged: particle spread sx={sx:.2f}, sy={sy:.2f}")
                break

        print("Main loop finished (iterations:", iteration, ")")

    finally:
        # Cleanup
        try:
            if showGUI:
                cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            if cam is not None and hasattr(cam, "terminateCaptureThread"):
                cam.terminateCaptureThread()
        except Exception:
            pass
        try:
            if arlo is not None and hasattr(arlo, "stop"):
                arlo.stop()
        except Exception:
            pass


if __name__ == "__main__":
    main()