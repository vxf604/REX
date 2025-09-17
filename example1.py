    # This script shows how to open a camera the picamera2 module and grab frames and show these.
    # Kim S. Pedersen, 2023

    import cv2  # Import the OpenCV library
    import time
    from pprint import *

    try:
        import picamera2

        print("Camera.py: Using picamera2 module")
    except ImportError:
        print("Camera.py: picamera2 module not available")
        exit(-1)


    print("OpenCV version = " + cv2.__version__)

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

    while True:
        image = cam.capture_array("main")
        cv2.imwrite("captured_image.png", image)  # Save the image
        print("Image saved as captured_image.png")
        break  # Exit after saving one image


    # Finished successfully
