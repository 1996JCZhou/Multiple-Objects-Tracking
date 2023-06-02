import cv2
import numpy as np


def draw_trace(img, trace_list, color):
    """
    Draw a trace on the current video frame.

    Args:
        img (Numpy array): Current video frame.
        trace_list (List): Updated trace list for the found target box center / the a-posteriori box center for the current video frame.  
    """

    for i, item in enumerate(trace_list):
        if i < 1:
            continue
        cv2.line(img,
                 (trace_list[i][0], trace_list[i][1]),
                 (trace_list[i-1][0], trace_list[i-1][1]),
                 color,
                 3)


def state2box(state):
    """
    

    Args:
        state [2D numpy array]: A BB position ("xywh").

    Returns:
        Output: _description_
    """

    center_x = state[0]
    center_y = state[1]
    w = state[2]
    h = state[3]
    return [int(i) for i in [center_x - w/2, center_y - h/2, center_x + w/2, center_y + h/2]]


def xyxy_to_xywh(xyxy):
    """
    Transform a bounding box's description using top left and bottom right pixel positions
    into the bounding box's description using box center position, width and height.

    Args:
        xyxy [1D numpy array]: [left top x coordinate, left top y coordinate, right bottom x coordinate, right bottom y coordinate].

    Returns:
        xywh [2D numpy array]: [[center x coordinate, center y coordinate, width, height]].T.
    """

    center_x = (xyxy[0] + xyxy[2]) / 2.0
    center_y = (xyxy[1] + xyxy[3]) / 2.0
    w = xyxy[2] - xyxy[0]
    h = xyxy[3] - xyxy[1]

    return np.array([[center_x, center_y, w, h]]).T


def xywh_to_state(xywh):
    """
    Extend the input detected BB position ("xywh") to state format.

    Args:
        mea [2D numpy array]: Detected BB position [[center x coordinate, center y coordinate, width, height]].T.

    Returns:
        Output [2D numpy array]: Extend the detected BB position to state format (with additional two elements).
                                 [[center x coordinate, center y coordinate, width, height, 0, 0]].T.
    """

    return np.row_stack((xywh, np.zeros((2, 1))))


def xywh_to_xyxy_int(obs):
    """
    Transform a bounding box's description using using box center position, width and height
    into the bounding box's description using top left and bottom right pixel positions.

    Args:
        obs [2D numpy array]: BB position with [[center x coordinate, center y coordinate, width, height]].T.

    Returns:     
        Output [List of integers]: [left top x coordinate, left top y coordinate, right bottom x coordinate, right bottom y coordinate].
    """

    center_x = obs[0]
    center_y = obs[1]
    w = obs[2]
    h = obs[3]

    x1 = center_x - w / 2.0
    y1 = center_y - h / 2.0
    x2 = center_x + w / 2.0
    y2 = center_y + h / 2.0

    return [int(i) for i in [x1, y1, x2, y2]]


def cal_iou(state, measure):
    """
    Calculate the IoU between two bounding boxes.

    Args:
        state   [2D numpy array]: BB position with [[center x coordinate, center y coordinate, width, height]].T.
        measure [2D numpy array]: BB position with [[center x coordinate, center y coordinate, width, height]].T.

    Returns:
        IoU (Float): Intersection over union.
    """

    """Transform a bounding box's description using box center position, width and height
       into the bounding box's description using top left and bottom right pixel positions."""
    state   = xywh_to_xyxy_int(state)
    measure = xywh_to_xyxy_int(measure)

    # The x coordinate of the left top pixel point of the right bounding box.
    xA = max(state[0], measure[0])
    # The x coordinate of the right bottom pixel point of the left bounding box.
    xB = min(state[2], measure[2])
    # The y coordinate of the left top pixel point of the bottom bounding box.
    yA = max(state[1], measure[1])
    # The y coordinate of the right bottom pixel point of the top bounding box.
    yB = min(state[3], measure[3])

    inter_h = xB - xA if xB - xA >= 1 else 0
    inter_w = yB - yA if yB - yA >= 1 else 0

    inter_area = inter_h * inter_w

    state_Area = (state[2] - state[0]) * (state[3] - state[1])
    measure_Area = (measure[2] - measure[0]) * (measure[3] - measure[1])

    IoU = inter_area / (state_Area + measure_Area - inter_area)

    return IoU
