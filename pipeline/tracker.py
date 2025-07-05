# Note: A typical tracker design implements a dedicated filter class for keeping the individual state of each track
# The filter class represents the current state of the track (predicted position, size, velocity) as well as additional information (track age, class, missing updates, etc..)
# The filter class is also responsible for assigning a unique ID to each newly formed track

import numpy as np

class Filter:
    """
    Kalman filter for tracking a single object in 2D.
    State: [x, y, vx, vy]
    Observation: [x, y]
    """

    def __init__(self, z, cls, dt=1.0):
        """
        Initialize filter with first measurement.

        Args:
            z (array-like): Detection [x_center, y_center, width, height].
            cls (any): Object class.
            dt (float): Time step.
        """
        self.dt = dt  # time step between updates
        self.cls = cls  # object class label
        self.id = None  # unique track ID, to be assigned later

        x, y = z[0], z[1]  # extract initial position from detection
        self.x = np.array([x, y, 0, 0], dtype=float)  # initial state vector [x, y, vx, vy]

        self.P = np.eye(4) * 500.0  # initial state covariance matrix with high uncertainty

        self.F = np.array([  # state transition matrix (constant velocity model)
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ])

        """
        State transition matrix F:
        Models the system dynamics assuming a constant velocity model.
        It updates the state vector [x, y, vx, vy] from the previous time step to the current,
        where position is updated by velocity multiplied by time step dt,
        and velocity remains constant.
        """

        self.H = np.array([  # observation matrix (we only observe position)
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])

        """ 
        Observation matrix H:
        Maps the full state vector [x, y, vx, vy] to the observed measurement space [x, y].
        Since only position (x, y) is directly measurable, H extracts these components.
        This means measurements correspond to the position part of the state,
        ignoring velocity components during the update step.
        """

        q = 1.0  # process noise scalar
        self.Q = q * np.array([  # process noise covariance matrix
            [dt**4/4,     0, dt**3/2,     0],
            [0,     dt**4/4,     0, dt**3/2],
            [dt**3/2,     0, dt**2,     0],
            [0, dt**3/2,     0, dt**2],
        ])

        """
        Process noise covariance matrix Q:
        Models the uncertainty in the system dynamics, assuming a constant velocity motion model.
        The entries depend on the time step dt and represent variance contributions from
        acceleration noise affecting position and velocity.
        The top-left 2x2 block relates to position noise,
        the bottom-right 2x2 block relates to velocity noise,
        and off-diagonal blocks capture correlation between position and velocity noise.
        Multiplying by scalar q adjusts the overall process noise strength.
        """

        r = 10.0  # measurement noise scalar
        self.R = np.eye(2) * r  # measurement noise covariance matrix

        self.box_w = z[2]  # store width of bounding box
        self.box_h = z[3]  # store height of bounding box

        self.track_age = 1  # number of total frames since initialization
        self.missed_frames = 0  # number of consecutive frames without update
        self.hits = 1  # number of successful updates
        self.is_confirmed = False  # track is confirmed after a few hits
    
    def predict(self, optical_flow):
        """
        Predict the next state and covariance of the Kalman filter.
        """
        cam_dx, cam_dy = optical_flow  # unpack camera movement offsets
        
        self.x = self.F @ self.x  # predict state: x_k = F * x_{k-1}
        self.x[0] += cam_dx  # compensate position x with camera motion
        self.x[1] += cam_dy  # compensate position y with camera motion

        self.P = self.F @ self.P @ self.F.T + self.Q  # predict covariance

        self.track_age += 1  # increase track age
        self.missed_frames += 1  # increase missed frames count (no measurement update yet)
    
    def update(self, z, optical_flow):
        """
        Update the Kalman filter state with a new measurement.
        """
        z_pos = np.array([z[0], z[1]])  # extract position from measurement

        y = z_pos - (self.H @ self.x)  # innovation (measurement residual)
        S = self.H @ self.P @ self.H.T + self.R  # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain

        self.x = self.x + K @ y  # update state estimate
        I = np.eye(4)  # identity matrix for covariance update
        self.P = (I - K @ self.H) @ self.P  # update covariance estimate

        self.box_w = z[2]  # update box width
        self.box_h = z[3]  # update box height

        self.missed_frames = 0  # reset missed frames
        self.track_age += 1  # increase track age

        if not self.is_confirmed:
            self.hits += 1  # increase hits (measurement updates)
            if self.hits >= 3:
                self.is_confirmed = True  # confirm track after 3 hits

    @property  # decorator: to allow access like an attribute without calling as a method
    def box(self):
        """
        Returns the current bounding box estimate.
        """
        return np.array([self.x[0], self.x[1], self.box_w, self.box_h])

    @property
    def velocity(self):
        """
        Returns the current velocity estimate.
        """
        return self.x[2:4]

class Tracker:
    def __init__(self):
        self.name = "Tracker"

    def start(self, data):
        # TODO: Implement start up procedure of the module
        pass

    def stop(self, data):
        # TODO: Implement shut down procedure of the module
        pass

    def step(self, data):
        # TODO: Implement processing of a detection list
        # The task of the tracker module is to identify (temporal) consistent tracks out of the given list of detections
        # The tracker maintains a list of known tracks which is initially empty.
        # The tracker then tries to associate all given detections from the detector to existing tracks. A meaningful metric needs to be defined
        # to decide which detection should be associated with each track and which detections better stay unassigned.
        # After the association step, one must handle there different cases:
        #   1) Detections which have not beed associated with a track: For these, create a new filter class and initialize its state based on the detection
        #   2) Tracks which have a detection: The state of these can be updated based on the associated detection
        #   3) Tracks which have no detection: It makes sense to allow for a few missing frames, nonetheless it is still necessary to predict the
        #      current filter state (e.g. based on the optical flow measurement and the object velocity). If too many frames are missing, the track can be deleted

        # Note: You can access data["detections"] and data["classes"] to receive the current list of detections and their corresponding classes
        # You must return a dictionary with the given fields:
        #       "tracks":           A Nx4 NumPy Array containing a 4-dimensional state vector for each track. Similar to the detections,
        #                           the track state containts the center point (X,Y) as well as the bounding box width and height (W, H)
        #       "trackVelocities":  A Nx2 NumPy Array with an additional velocity estimate (in pixels per frame) for each track
        #       "trackAge":         A Nx1 List with the track age (number of total frames this track exists). The track age starts at
        #                           1 on track creation and increases monotonically by 1 per frame until the track is deleted.
        #       "trackClasses":     A Nx1 List of classes associated with each track. Similar to detections, the following mapping must be used
        #                               0: Ball
        #                               1: GoalKeeper
        #                               2: Player
        #                               3: Referee
        #       "trackIds":         A Nx1 List of unique IDs for each track. IDs must not be reused and be unique during the lifetime of the program.
        return {
            "tracks": None,
            "trackVelocities": None,
            "trackAge": None,
            "trackClasses": None,
            "trackIds": None,
        }