"""Define file paths."""
FILE_DIR                     = "D:\\Object Detection\\MOT\\data\\labels"
VIDEO_PATH                   = "D:\\Object Detection\\MOT\\data\\testvideo.mp4"
VIDEO_OUTPUT_PATH_HUNGARIAN  = "D:\\Object Detection\\MOT\\data\\Kalman_Filter_MOT_Hungarian.avi"
VIDEO_OUTPUT_PATH_MAX_WEIGHT = "D:\\Object Detection\\MOT\\data\\Kalman_Filter_MOT_Max_Weight.avi"

"""Save the edited video frames as a video file."""
SAVE_VIDEO = True

"""Define the output video's displaying speed."""
FPS = 10

"""Define the threshold for minimal IOU value."""
IOU_MIN = 0.3

"""Define the tracking termination frame number"""
TERMINATE_FRAME = 5

"""Define the time intervall (ms) between two adjacent video frames."""
DELTA_T = 0.1

"""Color for display."""
GREEN = (0,   255, 0)
RED   = (0,   0,   255)
WHITE = (255, 255, 255)

"""Define the time intervall (ms) to display video frames."""
TIME_INTERVALL = 100
