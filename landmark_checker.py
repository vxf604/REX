import os
import cv2
import cv2.aruco as aruco


class LandmarkChecker:
    def __init__(self, landmark_radius=180, scale=10):
        self.landmark_radius = landmark_radius
        self.SCALE = scale

    def checkForLandmark(self, cam):
        os.makedirs("images", exist_ok=True)

        image = cam.capture()
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
        Z = cam.f * cam.X / x
        Z_scaled = Z / self.SCALE
        print(f"Distance to landmark Z: {Z_scaled}")
        return True, corners, ids, Z_scaled
