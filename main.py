import cv2, const, utils, measure, os
import numpy as np
from time    import time
from kalman  import Kalman_Filter_Tracker

"""Preparation for the Kalman Filter.
There are two reasons not to directly link the observations from frame to frame.
1. There will be no detected bounding box when the target is occluded.
2. The observations are still noisy even from a good detector."""
## -----------------------------------------------------------------------------------------
## -----------------------------------------------------------------------------------------
"""Define the system model."""
# Assume that the target object is moving in a straight line with a uniform velocity.
# System matrix.
# DELTA_T: The time intervall (s) between two adjacent video frames. 
A = np.array([[1, 0, 0, 0, const.DELTA_T, 0            ],  # Bounding box center pixel x coordinate, Xt+1 = Xt + Vx,t * DELTA_T.
              [0, 1, 0, 0, 0,             const.DELTA_T],  # Bounding box center pixel y coordinate, Yt+1 = Yt + Vy,t * DELTA_T.
              [0, 0, 1, 0, 0,             0            ],  # Bounding box width                    , Ht+1 = Ht.
              [0, 0, 0, 1, 0,             0            ],  # Bounding box height                   , Wt+1 = Wt.
              [0, 0, 0, 0, 1,             0            ],  # Bounding box center pixel x velocity  , Vx,t+1 = Vx,t.
              [0, 0, 0, 0, 0,             1            ]]) # Bounding box center pixel y velocity  , Vy,t+1 = Vy,t.

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
R = np.eye(6) * 0.1

"""Initialize the covariance matrix of the A-posteriori-Density.
   (If unknown, then asign it towards infinity.)"""
P = np.eye(6) * 2000
## -----------------------------------------------------------------------------------------
## -----------------------------------------------------------------------------------------


def main():
## -----------------------------------------------------------------------------------------
## -----------------------------------------------------------------------------------------
    """Load video and all the detected Bounding Box (BB) positions in each video frame."""
    assert os.path.exists(const.VIDEO_PATH), "Path for video does not exist."
    cap = cv2.VideoCapture(const.VIDEO_PATH)

    assert os.path.exists(const.FILE_DIR), "Path for labels does not exist."
    obs_list_all_frames = measure.load_observations(const.FILE_DIR)
    # 'obs_list_all_frames': A list of elements.
    # Every element in this list is also a list corresponding to a single video frame.
    # This list consists of all the detected BB positions in each video frame.
    # Each BB position is described as an 1D numpy array ("xyxy").
## -----------------------------------------------------------------------------------------
## -----------------------------------------------------------------------------------------
    """Aggregate edited video frames into a new video for presentation."""
    if const.SAVE_VIDEO:

        """Define the video encoder."""
        # Encode video or audio into a specific file format
        # according to the encoding format using an encoder 'fourcc'.
        fourcc = cv2.VideoWriter_fourcc(*'XVID')

        """Define the size of the new video."""
        sz = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
              int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        """Define a path to save the new video."""
        assert os.path.exists(const.VIDEO_OUTPUT_PATH_MAX_WEIGHT), "Path to save video does not exist."
        save_path = const.VIDEO_OUTPUT_PATH_MAX_WEIGHT

        out = cv2.VideoWriter(save_path, fourcc, const.FPS_OUT, sz, isColor=True)
## -----------------------------------------------------------------------------------------
## -----------------------------------------------------------------------------------------
    """Initialize a list to save kalman filter instances
       for each target appearing in each video frame."""
    kalman_list = [] 

    """Count the video frames for display."""
    frame_count = 1

    """Load all the detected BB positions (1D numpy array; "xyxy") for each video frame."""
    for obs_list_frame in obs_list_all_frames:

        """Load the each video frame for display."""
        ret, frame = cap.read()

        if not ret:
            break

        """Begin to record the calculation time."""
        t0 = time()
    ## -----------------------------------------------------------------------------------------
    ## -----------------------------------------------------------------------------------------
        """For each kalman filter instance:
           1. Process the prediction step
              to get its expected value vector of the A-prior-Density ('X_prior')
              for the current video frame.
           2. Extract the predicted BB position (2D numpy array; "xywh")
              and save it in a list 'state_list'."""
        state_list = []
        for kalman in kalman_list:        # For each instance 'kalman' in the 'kalman_list':
            kalman.predict()              # Process the prediction step in the current video frame.
            state = kalman.X_prior        # Expected value vector of the A-prior-Density in the current video frame.
            state_list.append(state[0:4]) # Predicted BB position (2D numpy array; "xywh") extracted from the state.

        """Load all the detected BB positions in the current video frame.
           Transform them from 1D numpy arrays ("xyxy") into 2D numpy arrays ("xywh")."""
        # 'obs_list': A list of each detected BB position (2D numpy array; "xywh")
        #             in the current video frame.
        #             [2D numpy array ("xywh"), 2D numpy array ("xywh"), ...]
        obs_list = [utils.xyxy_to_xywh(obs) for obs in obs_list_frame]

        """Matching between predicted and detected BB positions in the current video frame
           to get a list of unmatched states and observations and matched pairs."""
        state_unmatched_list, obs_unmatched_list, matched_pairs, matched_list = Kalman_Filter_Tracker.Matching(state_list, obs_list)
    ## -----------------------------------------------------------------------------------------
    ## -----------------------------------------------------------------------------------------
        """To deal with matched pairs."""
        if matched_pairs:
            for matched_pair in matched_pairs:

                """Key component for tracking:
                   Matching between predicted and detected BB positions in the current video frame."""
                """
                In the current video frame:
                (The detected BB 'obs_list[matched_pair[1]]') matches
                (the predicted BB from the kalman filter instance 'kalman_list[matched_pair[0]]'
                 after the prediction step).
                -->> This detected BB 'obs_list[matched_pair[1]]'
                     is determined to be an observation for the current video frame.
                -->> Process the filter step of the kalman filter instance
                     using this determined observation to calculate the filtered result
                     als preparation for the next video frame.
                """
                kalman_list[matched_pair[0]].update(obs_list[matched_pair[1]])

        """To deal with unmatched states (e.g. occluded targets)."""
        state_del = []
        for idx in state_unmatched_list:

            """No detection text reminder."""
            cv2.putText(frame,                                                         # Current video frame, where we put on the text.
                        "Lost",                                                        # Text.
                        (utils.xywh_to_xyxy_int(kalman_list[idx].X_prior[:4])[0], \
                         utils.xywh_to_xyxy_int(kalman_list[idx].X_prior[:4])[1] - 5), # Position of the text.
                        cv2.FONT_HERSHEY_SIMPLEX,                                      # Type of the text.
                        0.7,                                                           # Size of the text.
                        (255, 0, 0),                                                   # Blue text.
                        1)                                                             # Thickness of the text.

            """If the observation for the current time step is not available,
               which means the detector has lost its target,
               then we skip the filter step, beacause the kalman gain is zero."""
            status, _, _ = kalman_list[idx].update()

            """Once the detector has lost the target for 'TERMINATE_FRAME' times (normally in a row),
               then we delete this target's kalman filter instance and will no longer track this target."""
            if not status:
                state_del.append(idx)

        """Update the list for all the instantiated kalman filters."""
        kalman_list = [kalman_list[i] for i in range(len(kalman_list)) if i not in state_del]

        """To deal with unmatched detections."""
        """For each unmatched detection
           (e.g. new appearing target / target moving fast between adjacent video frames):
           initialize a kalman filter instance by using this unmatched detection BB 'obs_list[idx]'
           as the initial expected value vector of the A-posteriori-Density 'X_posteriori'
           in the current video frame.
        """
        for idx in obs_unmatched_list:

            """Apply no detection text reminder."""
            cv2.putText(frame,                                          # Current video frame, where we put on the text.
                        "New",                                          # Text.
                        (utils.xywh_to_xyxy_int(obs_list[idx])[0], \
                         utils.xywh_to_xyxy_int(obs_list[idx])[1] - 5), # Position of the text.
                        cv2.FONT_HERSHEY_SIMPLEX,                       # Type of the text.
                        0.7,                                            # Size of the text.
                        (0, 0, 255),                                    # Red text.
                        1)                                              # Thickness of the text.

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
        # A matched pair consists of a predicted BB position and a detected BB position in the current video frame.
        for item in matched_list:
            cv2.line(frame, tuple(item[0][:2]), tuple(item[1][:2]), const.WHITE, 3, cv2.LINE_AA)
        # In the first video frame, there is no matched pair (white).

        """Display the trace of each kalman filter instance."""
        for kalman in kalman_list:
            trace_list = kalman.track
            utils.draw_trace(frame, trace_list, kalman.track_color)

        """Display title."""
        cv2.putText(frame, "ALL DETECTED BOXES (Green)",            (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(frame, "CURRENT BEST ESTIMATION BOXES (Red)",   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
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
