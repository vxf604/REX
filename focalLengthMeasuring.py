import cv2
import cv2.aruco as aruco
import time
from pprint import *
import os
import datetime
import sys

try:
    import picamera2  # type: ignore

    print("Camera.py: Using picamera2 module")
except ImportError:
    print("Camera.py: picamera2 module not available")
    exit(-1)


print("OpenCV version = " + cv2.__version__)


def measureFocal(Z, X):
    # Open a camera device for capturing
    imageSize = (1640, 1232)
    FPS = 30
    cam = picamera2.Picamera2()
    frame_duration_limit = int(1 / FPS * 1000000)  # Microseconds
    # Change configuration to set resolution, framerate
    picam2_config = cam.create_video_configuration(
        {"size": imageSize, "format": "RGB888"},
        controls={
            "FrameDurationLimits": (frame_duration_limit, frame_duration_limit),
            "ScalerCrop": (0, 0, 3280, 2464),
        },
        queue=False,
    )
    cam.configure(picam2_config)  # Not really necessary
    cam.start(show_preview=False)

    pprint(cam.camera_configuration())  # Print the camera configuration in use
    time.sleep(1)  # wait for camera to setup

    image = cam.capture_array("main")
    os.makedirs("images", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = os.path.join("image", f"captured_image_{timestamp}.png")
    cv2.imwrite(image_path, image)

    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()
    corners, ids, rejected = aruco.detectMarkers(
        image, aruco_dict, parameters=parameters
    )
    if ids is None:
        print("No marker detected!")
        return

    c = corners[0][0]  # first marker detected
    pixel_width = int(cv2.norm(c[0] - c[1]))
    print("Z = Distance to marker (mm):", Z)
    print("X = Landmark width (mm):", X)
    print("x = Marker pixel width:", pixel_width)
    print("f = Focal length (pixels):", (pixel_width * Z) / X)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Provide the distance from camera to marker Z (mm)")
        exit(1)
    Z = float(sys.argv[1])
    X = 145
    measureFocal(Z, X)
