import random, const, utils
import numpy as np
from matcher import max_weight_matching, hungarian_algorithm


class Kalman_Filter_MOT:
    def __init__(self, A, B, C, L, Q, R, X, P):

        """Define the system model."""
        self.A = A  # System matrix.
        self.B = B  # Control matrix.
        self.C = C  # Observation matrix.
        self.L = L  # System noise matrix.

        """Define the noise covariance matrix."""
        self.Q = Q  # System noise covariance matrix.
        self.R = R  # Measurement noise covariance matrix.

        """Initialization for the Kalman Filter."""
        # 'X_posteriori': Expected value vector of the A-posteriori-Density as initialization.
        # 'P_posteriori': Covariance matrix     of the A-posteriori-Density as initialization.
        self.X_posteriori = X
        self.P_posteriori = P

        self.X_prior = None # Expected value vector of the A-prior-Density.
        self.P_prior = None # Covariance matrix     of the A-prior-Density.

        self.K = None  # Kalman gain.

        self.Z = np.zeros((6, 1))  # self.Z [2D numpy array]: Observation array [[BB's center pixel x coordinate,
                                   #                                              BB's center pixel y coordinate,
                                   #                                              BB's width,
                                   #                                              BB's height,
                                   #                                              BB's center pixel x velocity,
                                   #                                              BB's center pixel y velocity]].

        """Define the termination criterion."""
        self.terminate_count = const.TERMINATE_FRAME

        """Initialize a trace list 'self.track' for tracking
           and a random drawing color for each instance."""
        self.track = []
        self.track_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        """Update the trace list with initialized 'X_posteriori' and 'P_posteriori'."""
        self.updata_trace_list()


    def predict(self):
        """Process the prediction step."""
        # 'X_posteriori': Expected value vector of the A-posteriori-Density for the previous time step n.
        # 'P_posteriori': Covariance matrix     of the A-posteriori-Density for the previous time step n.
        # 'X_prior':      Expected value vector of the A-prior-Density for the current time step n+1.
        # 'P_prior':      Covariance matrix     of the A-prior-Density for the current time step n+1.
        self.X_prior = np.dot(self.A, self.X_posteriori)
        self.P_prior = np.dot(np.dot(self.A, self.P_posteriori), self.A.T) + np.dot(np.dot(self.L, self.Q), self.L.T)
        return self.X_prior, self.P_prior


    @staticmethod
    def Matching(state_list, obs_list):
        """
        Employ the 'Max Weight Matching' algorithm or the 'Hungarian' algorithm for matching.

        Args:
            state_list  [List]: A list of each predicted BB position (2D numpy array; "xywh") in the current video frame.
            obs_list    [List]: A list of each  detected BB position (2D numpy array; "xywh") in the current video frame.

        Returns:
            list(state_rec - state_used) [List]: A list of indice of unmatched states.
            list(obs_rec - obs_used)     [List]: A list of indice of unmatched observations.
            matched_pairs                [Set]:  A set of matched pairs of states and observations.
            matched_list                 [List]: A list of matched pairs of states and observations.
                                                 (BB's description using top left and bottom right pixel position.) 
        """

        """Process the 'Max Weight Matching' algorithm or the 'Hungarian' algorithm for matching."""
        # match_dict = hungarian_algorithm(state_list, obs_list)
        match_dict = max_weight_matching(state_list, obs_list)
    ## -----------------------------------------------------------------------------------------
    ## -----------------------------------------------------------------------------------------
        """Documentation."""

        """Document the states and observations indice that need to be matched."""
        state_rec = {i for i in range(len(state_list))}
        obs_rec =   {i for i in range(len(obs_list))}

        """Document the states and observations that have been matches."""
        state_used, obs_used, matched_pairs = set(), set(), set()

        """Preparation for displaying matched pairs of states and observations."""
        matched_list = []
    ## -----------------------------------------------------------------------------------------
    ## -----------------------------------------------------------------------------------------
        """Read matched pairs of states and observations."""
        for state, obs in match_dict.items():
            state_index = int(state.split('_')[1]) # 'state': 'state_%d' -->> 'state.split('_')': ['state', '%d'].
            obs_index   = int(obs.split('_')[1])   # 'obs':   'obs_%d'.  -->> 'obs.split('_')':   ['obs', '%d'].

            """Documentation for matched pairs."""
            state_used.add(state_index)
            obs_used.add(obs_index)
            matched_pairs.add((state_index, obs_index))
            matched_list.append([utils.xywh_to_xyxy_int(state_list[state_index]), utils.xywh_to_xyxy_int(obs_list[obs_index])])
    ## -----------------------------------------------------------------------------------------
    ## -----------------------------------------------------------------------------------------
        """Return unmatched indice of states and observations and matched pairs."""
        return list(state_rec - state_used), list(obs_rec - obs_used), matched_pairs, matched_list


    def update(self, obs=None):

        status = True

        """If observation for the current video frame is available,
           then process the filter step."""
        if obs is not None:

            """Compute the kalman gain."""
            # 'P_prior': Covariance matrix of the A-prior-density for the current video frame (time step n+1).
            k1 = np.dot(self.P_prior, self.C.T)
            k2 = np.dot(np.dot(self.C, self.P_prior), self.C.T) + self.R
            self.K = np.dot(k1, np.linalg.inv(k2))

            """Update the observation for the current time step."""
            self.Z[ :4] = obs

            """Calculate the center pixel velocity using the observation for the current video frame and
               the expected value vector of the A-posteriori-Density for the previous video frame / best estimation."""
            # 'X_posteriori': Expected value vector of the A-posteriori-density for the previous video frame.
            dx = (1 / const.DELTA_T) * (obs[0] - self.X_posteriori[0])
            dy = (1 / const.DELTA_T) * (obs[1] - self.X_posteriori[1])
            self.Z[4: ] = np.array([[dx, dy]]).T

            """Compute 'X_posteriori' and 'P_posteriori'."""
            # 'X_prior':      Expected value vector of the A-prior-density for the current video frame (time step n+1).
            # 'P_prior':      Covariance matrix     of the A-prior-density for the current video frame (time step n+1).
            # 'X_posteriori': Expected value vector of the A-posteriori-density for the current video frame (time step n+1).
            # 'P_posteriori': Covariance matrix     of the A-posteriori-density for the current video frame (time step n+1).
            self.X_posteriori = self.X_prior + np.dot(self.K, self.Z - np.dot(self.C, self.X_prior))
            self.P_posteriori = np.dot(np.eye(6) - np.dot(self.K, self.C), self.P_prior)

            """Update the trace list with computed 'X_posteriori' and 'P_posteriori'."""
            status = True

        else:
            # Once the detector has lost the target for 'TERMINATE_FRAME' times (normally in a row),
            # then we refuse to update the trace list for this target.
            if self.terminate_count == 0:
                status = False

            # If observation for the current time step is not available,
            # which means the detector has lost the target,
            # then the kalman gain is zero.
            else:
                self.terminate_count -= 1

                """Compute 'X_posteriori' and 'P_posteriori'."""
                # 'X_prior':      Expected value vector of the A-prior-density for the current video frame (time step n+1).
                # 'P_prior':      Covariance matrix     of the A-prior-density for the current video frame (time step n+1).
                # 'X_posteriori': Expected value vector of the A-posteriori-density for the current video frame (time step n+1).
                # 'P_posteriori': Covariance matrix     of the A-posteriori-density for the current video frame (time step n+1).
                self.X_posteriori = self.X_prior
                self.P_posteriori = self.P_prior

        """Update the trace list."""
        if status:
            self.updata_trace_list()

        return status, self.X_posteriori, self.P_posteriori


    """Key component for tracking."""
    def updata_trace_list(self, max_list_len=25):
        if len(self.track) <= max_list_len:
            # 'X_posteriori[0]': Best estimated / expected BB center pixel x coordinate of the A-posteriori-density for the current video frame.
            # 'X_posteriori[1]': Best estimated / expected BB center pixel y coordinate of the A-posteriori-density for the current video frame.
            self.track.append([int(self.X_posteriori[0]), int(self.X_posteriori[1])])
        else:
            self.track.pop(0) # Pop the first 'box_center' from the 'trace_list'.
            self.track.append([int(self.X_posteriori[0]), int(self.X_posteriori[1])])
