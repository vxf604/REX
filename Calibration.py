import cv2
import time
from pprint import *

imageSize = (1640, 1232)
FPS = 30
cam = picamera2.Picamera2()
frame_microseconds = int(1/FPS * 1000000)
picam2_config = cam.create_video_configuration({"size": imageSize, "format": 'RGB888'},
                                                            controls={"FrameDurationLimits": (frame_microseconds, frame_microseconds),
                                                            "ScalerCrop": (0,0,3280,2464)},
                                                            queue=False)
cam.start(show_preview=False)

print (cam.camera_configuration())
time.sleep(1)

WIN_RF = "Example 1"
cv2.namedWindow(WIN_RF)
cv2.moveWindow(WIN_RF, 100, 100)


while cv2.waitKey(4) == -1: 
    image = cam.capture_array("main")
    cv2.imshow(WIN_RF, image)

