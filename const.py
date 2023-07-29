import cv2

"""Define file paths."""
FILE_DIR                     = "D:\\Object Detection\\MOT\\data\\labels"
VIDEO_PATH                   = "D:\\Object Detection\\MOT\\data\\testvideo.mp4"
VIDEO_OUTPUT_PATH_HUNGARIAN  = "D:\\Object Detection\\MOT\\data\\Kalman_Filter_MOT_Hungarian.avi"
VIDEO_OUTPUT_PATH_MAX_WEIGHT = "D:\\Object Detection\\MOT\\data\\Kalman_Filter_MOT_Max_Weight.avi"

"""Save the edited video frames as a video file."""
SAVE_VIDEO = True

"""Define the output video's displaying speed."""
FPS_OUT = 10

"""Define the threshold for minimal IOU value."""
IOU_MIN = 0.3

"""Define the tracking termination frame number"""
TERMINATE_FRAME = 5

"""Calculate the time intervall (ms) between two adjacent video frames.
   The variable 'DELTA_T' is the reciprocal of the video FPS value."""

video = cv2.VideoCapture(VIDEO_PATH)

fps = video.get(cv2.CAP_PROP_FPS)
print("Frames per second: {}".format(fps))

frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
print("Total number of video frames: {}".format(frame_count))

DELTA_T = 1 / fps

"""Color for display."""
GREEN = (0,   255, 0)
RED   = (0,   0,   255)
WHITE = (255, 255, 255)

"""Define the time intervall (ms) to display video frames."""
TIME_INTERVALL = 100
