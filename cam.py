import numpy as np
import cv2
import cv2.aruco as aruco


class Cam:
    def __init__(self):
        try:
            import picamera2  # type: ignore

            print("Camera.py: Using picamera2 module")
        except ImportError:
            print("Camera.py: picamera2 module not available")
            exit(-1)
        self.f = 1226.11  # pixels
        self.X = 145  # mm
        self.cWidth = 1640
        self.cHeight = 1232
        self.cx = self.cWidth / 2
        self.cy = self.cHeight / 2
        self.imageSize = (self.cWidth, self.cHeight)
        self.intrinsicMatrix = np.array(
            [[self.f, 0, self.cx], [0, self.f, self.cy], [0, 0, 1]], dtype=np.float32
        )
        self.distCoeffs = np.zeros((5, 1))
        self.FPS = 30
        self.picam = picamera2.Picamera2()
        self.frame_duration_limit = int(1 / self.FPS * 1000000)
        self.picam2_config = cam.create_video_configuration(
            {"size": self.imageSize, "format": "RGB888"},
            controls={
                "FrameDurationLimits": (
                    self.frame_duration_limit,
                    self.frame_duration_limit,
                ),
                "ScalerCrop": (0, 0, 3280, 2464),
            },
            queue=False,
        )

        self.configure(self.picam2_config)
        self.start()

    def configure(self, config):
        self.picam2.configure(config)

    def start(self, show_preview=False):
        self.picam2.start(show_preview=show_preview)

    def stop(self):
        self.picam2.stop()

    def capture(self, request):
        return self.cam.capture_array("main")

    def estimatePose(self, corners):
        rvecs, tvecs, objPoints = cv2.aruco.estimatePoseSingleMarkers(
            corners,
            self.X,
            self.intrinsicMatrix,
            self.distCoeffs,
        )
        return rvecs, tvecs, objPoints
