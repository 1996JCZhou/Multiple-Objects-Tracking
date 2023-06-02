import cv2, const, utils, measure, os
import numpy as np
from time    import time
from kalman  import Kalman_Filter_Tracker

"""Preparation for the Kalman Filter.
There are two reasons not to directly link the observations from frame to frame.
1. There will be no detected bounding box when the target is occluded.
2. The observations are still noisy even from good detecotr."""

"""Define the system model."""
# System matrix.
A = np.array([[1, 0, 0, 0, const.DELTA_T, 0            ],  # center pixel x coordinate
              [0, 1, 0, 0, 0,             const.DELTA_T],  # center pixel y coordinate
              [0, 0, 1, 0, 0,             0            ],  # box width
              [0, 0, 0, 1, 0,             0            ],  # box height
              [0, 0, 0, 0, 1,             0            ],  # center pixel x velocity
              [0, 0, 0, 0, 0,             1            ]]) # center pixel y velocity

# Control matrix.
B = None

# Observation matrix.
C = np.eye(6)

# System noise matrix.
# System noise here comes from uncertainties in target movement,
# e.g. sudden acceleration, deceleration, turns, etc.
L = np.eye(6)

"""Define the noise covariance matrix."""
# System noise covariance matrix.
Q = np.eye(6) * 0.1

# Measurement noise covariance matrix.
# Although the results from the detector are realible, they are still noisy.
R = np.eye(6)

"""Define the covariance matrix of the A-posteriori-Density
   for the time step 0. (unknown, then asigned towards infinity)"""
P = np.eye(6) * 2000


def main():
## -----------------------------------------------------------------------------------------
## -----------------------------------------------------------------------------------------
    """Load video and detected BB positions for each video frame."""
    assert os.path.exists(const.VIDEO_PATH), "Path for video does not exist."
    cap = cv2.VideoCapture(const.VIDEO_PATH)

    assert os.path.exists(const.FILE_DIR), "Path for labels does not exist."
    obs_list_all_frames = measure.load_observations(const.FILE_DIR)
    # 'obs_list_all_frames': A list of elements.
    # Every element of this list is also a list corresponding to a video frame.
    # This list consists of all the detected bounding box positions in each video frame.
    # Each BB position is described as an 1D numpy array ("xyxy").

    """Save the edited video frames as a video file."""
    if const.SAVE_VIDEO:

        """Define the video encoder."""
        # Encoding video or audio into a specific file format
        # according to the encoding format requires an encoder 'fourcc'.
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        sz = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
              int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        assert os.path.exists(const.VIDEO_OUTPUT_PATH_MAX_WEIGHT), "Path to save video does not exist."
        save_path = const.VIDEO_OUTPUT_PATH_MAX_WEIGHT

        """Make a video using edited video frames."""
        out = cv2.VideoWriter(save_path, fourcc, const.FPS, sz, isColor=True)
## -----------------------------------------------------------------------------------------
## -----------------------------------------------------------------------------------------
    """Initialize a list of kalman filter instances
       for the appeared target in the video frame."""
    kalman_list = [] 

    """Load all the detected BB positions (1D numpy array; "xyxy")
       in the current video frame."""
    frame_count = 1
    for obs_list_frame in obs_list_all_frames:

        """Load the corresponding video frame for display."""
        ret, frame = cap.read()
        if not ret:
            break
    ## ------------------------------------------------------------------------------------
    ## ------------------------------------------------------------------------------------
        """Begin to record calculation time."""
        t0 = time()
    ## -----------------------------------------------------------------------------------------
    ## -----------------------------------------------------------------------------------------
        """For each kalman filter instance:
           1. Process the prediction step in the current video frame
              to get its expected value vector of the A-prior-Density ('X_prior').
           2. Extract the predicted BB position (2D numpy array; "xywh")
              and save it in a list."""
        state_list = list()
        for kalman in kalman_list:        # For each instance 'kalman' in the 'kalman_list'.
            kalman.predict()              # Process the prediction step in the current video frame.
            state = kalman.X_prior        # Expected value vector of the A-prior-Density in the current time step.
            state_list.append(state[0:4]) # Extracted predicted BB position (2D numpy array; "xywh").

        """Load all the detected BB positions in the current video frame.
           Transform them into 2D numpy arrays ("xywh")."""
        # 'obs_list': A list of each detected BB position (2D numpy array; "xywh")
        #             in the current video frame.
        # 'obs_list' = [2D numpy array ("xywh"), 2D numpy array ("xywh"), ...]
        obs_list = [utils.xyxy_to_xywh(obs) for obs in obs_list_frame]
    ## -----------------------------------------------------------------------------------------
    ## -----------------------------------------------------------------------------------------
        """Association between predicted and detected BB positions in the current video frame.
           Update each matched kalman filter instance using its matched detected BB position as observation."""
        state_unmatched_list, obs_unmatched_list, matched_pairs, matched_list = Kalman_Filter_Tracker.association(state_list, obs_list)
    ## -----------------------------------------------------------------------------------------
    ## -----------------------------------------------------------------------------------------
        """To deal with matched pairs."""
        if matched_pairs:
            for matched_pair in matched_pairs:

                """Key component for tracking."""
                """In the current video frame:
                (The detected BB 'obs_list[obs_index]') matches
                (the predicted BB from the kalman filter instance 'kalman_list[state_index]' for the target
                after the prediction step).
                -->> This detected BB 'obs_list[obs_index]' is determined to be an observation.
                -->> Process the filter step of the kalman filter instance using this observation
                     to prepare for the next video frame."""
                kalman_list[matched_pair[0]].update(obs_list[matched_pair[1]])
    ## -----------------------------------------------------------------------------------------
        """To deal with unmatched states (e.g. occluded targets)."""
        state_del = list()
        for idx in state_unmatched_list:

            """No detection reminder."""
            cv2.putText(frame,                                                         # Current video frame, where we put on the text.
                        "Lost",                                                        # Text.
                        (utils.xywh_to_xyxy_int(kalman_list[idx].X_prior[:4])[0], \
                         utils.xywh_to_xyxy_int(kalman_list[idx].X_prior[:4])[1] - 5), # Position of the text.
                        cv2.FONT_HERSHEY_SIMPLEX,                                      # Type of the text.
                        0.7,                                                           # Size of the text.
                        (255, 0, 0),                                                   # Blue text.
                        1)                                                             # Thickness of the text.

            """If the observation for the current time step is not available,
               which means the detector has lost the target,
               then we skip the filter step, beacause the kalman gain is zero."""
            status, _, _ = kalman_list[idx].update()

            """Once the detector has lost the target for (TERMINATE_FRAME - 1) times (normally in a row),
               then we delete this target's kalman filter instance and will no longer track this target."""
            if not status:
                state_del.append(idx)

        """Update the list for all the instantiated kalman filters."""
        kalman_list = [kalman_list[i] for i in range(len(kalman_list)) if i not in state_del]
    ## -----------------------------------------------------------------------------------------
        """To deal with unmatched detections."""
        """Initialize a kalman filter instance (with unmatched detection)
           for each target (e.g. new appeared / fast moving between adjacent video frames),
           to which the unmatched detection refers to, using unmatched detection 'obs_list[idx]'."""
        for idx in obs_unmatched_list:
    
            """No detection reminder."""
            cv2.putText(frame,                                                         # Current video frame, where we put on the text.
                        "New",                                                        # Text.
                        (utils.xywh_to_xyxy_int(obs_list[idx])[0], \
                         utils.xywh_to_xyxy_int(obs_list[idx])[1] - 5), # Position of the text.
                        cv2.FONT_HERSHEY_SIMPLEX,                                      # Type of the text.
                        0.7,                                                           # Size of the text.
                        (0, 0, 255),                                                   # Red text.
                        1)                                                             # Thickness of the text.

            kalman_list.append(Kalman_Filter_Tracker(A, B, C, L, Q, R, utils.xywh_to_state(obs_list[idx]), P))
## -----------------------------------------------------------------------------------------
## -----------------------------------------------------------------------------------------
        """"Visualisation"""
        """Display all the detected BB in the current video frame (Green)."""
        for obs in obs_list_frame:
            cv2.rectangle(frame, tuple(obs[:2]), tuple(obs[2:]), const.GREEN, 1, cv2.LINE_AA)

        """Display all the best estimated BB in the current video frame (Red)."""
        for kalman in kalman_list:
            pos = utils.xywh_to_xyxy_int(kalman.X_posteriori)
            cv2.rectangle(frame, tuple(pos[:2]), tuple(pos[2:]), const.RED, 1, cv2.LINE_AA)

        """Display the matched pairs (White)."""
        for item in matched_list:
            cv2.line(frame, tuple(item[0][:2]), tuple(item[1][:2]), const.WHITE, 3, cv2.LINE_AA)

        """Display the trace of each kalman filter instance."""
        for kalman in kalman_list:
            trace_list = kalman.track
            utils.draw_trace(frame, trace_list, kalman.track_color)

        """Display title."""
        cv2.putText(frame, "ALL BOXES(Green)",                      (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(frame, "CURRENT BEST ESTIMATION BOXES(Red)",    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(frame, "CURRENT FRAME: {}".format(frame_count), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        frame_count += 1

        """Display the current video frame and all the drawings on it."""
        # Refresh the window "track" with the current video frame
        # and all the drawing on the current video frame,
        # neglecting all the drawing on the previous video frames.
        # (different from the static image).
        cv2.imshow("track", frame)
    ## ------------------------------------------------------------------------------------
    ## ------------------------------------------------------------------------------------
        """Save the current edited video frame."""
        if const.SAVE_VIDEO:
            out.write(frame)
    ## ------------------------------------------------------------------------------------
    ## ------------------------------------------------------------------------------------
        """Begin to record calculation time."""
        t1 = time()
        print(f"Duration for calculation is: {t1 - t0}.")
    ## ------------------------------------------------------------------------------------
    ## ------------------------------------------------------------------------------------
        """Press the 'Esc' or the 'q' key to immediately exit the program."""
        # The ASCII code value corresponding to the same key on the keyboard in different situations
        # (e.q. when the key "NumLock" is activated)
        # is not necessarily the same, and does not necessarily have only 8 bits, but the last 8 bits must be the same.
        # In order to avoid this situation, quote &0xff, to get the last 8 bits of the ASCII value of the pressed key
        # to determine what the key is.
        c = cv2.waitKey(const.TIME_INTERVALL) & 0xFF # Wait for 10 ms.
        if c == 27 or c == ord('q'):
            break
## -----------------------------------------------------------------------------------------
## -----------------------------------------------------------------------------------------
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
