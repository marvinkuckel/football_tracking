# Note: A typical tracker design implements a dedicated filter class for keeping the individual state of each track
# The filter class represents the current state of the track (predicted position, size, velocity) as well as additional information (track age, class, missing updates, etc..)
# The filter class is also responsible for assigning a unique ID to each newly formed track

import numpy as np
from scipy.optimize import linear_sum_assignment  # for solving the assignment problem

class Filter:
    def __init__(self, z, cls):
        """
        Initialize filter with first measurement.

        Args:
            z (array-like): Detection [x_center, y_center, width, height].
            cls (any): Object class.
            dt (float): Time step.
        """
        self.id = None  # unique track ID, to be assigned later
        self.cls = cls  # object class label
        
        x, y, w, h = z  # extract initial position and size from detection
        self.x = np.array([x, y, 0, 0, 0, 0, w, h], dtype=float)  # initial state vector [x, y, vx, vy, ax, ay, w, h]

        self.P = np.diag([5.0, 5.0, 50.0, 59.0, 100.0, 100.0, 5.0, 5.0])  # initial state covariance matrix

        self.F = np.array([  # state transition matrix (constant velocity model (+ width & height))
            [1, 0, 1, 0, 1/2, 0,   0, 0],
            [0, 1, 0, 1, 0,   1/2, 0, 0],
            [0, 0, 1, 0, 0,   0,   0, 0],
            [0, 0, 0, 1, 0,   0,   0, 0],
            [0, 0, 0, 0, 1,   0,   0, 0],
            [0, 0, 0, 0, 0,   1,   0, 0],
            [0, 0, 0, 0, 0,   0,   1, 0],
            [0, 0, 0, 0, 0,   0,   0, 1],
        ])

        """
        State transition matrix F:
        Models the system dynamics assuming a constant velocity model.
        It updates the state vector [x, y, vx, vy, w, h] from the previous time step to the current,
        where position is updated by velocity multiplied by time step dt,
        and velocity as well as size remains constant.
        """

        self.H = np.array([  # observation matrix (we only observe position and size)
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ])

        """ 
        Observation matrix H:
        Maps the full state vector [x, y, vx, vy, w, h] to the observed measurement space [x, y, w, h].
        Since only position (x, y) and size (w, h) is directly measurable, H extracts these components.
        This means measurements correspond to the position and size part of the state,
        ignoring velocity components during the update step.
        """

        # if object is a ball, give it higher uncertainty
        if cls == 0:
            q_pos = 20.0
            q_vel = 15.0
            q_acc = 10.0
            q_box = 5.0
        else: 
            q_pos = 8.0
            q_vel = 4.0
            q_acc = 2.0
            q_box = 0.2
            
        self.Q = np.array([  # process noise covariance matrix
            [q_pos/20, 0,        q_pos/8, 0,       q_pos/6, 0,       0,     0],
            [0,        q_pos/20, 0,       q_pos/8, 0,       q_pos/6, 0,     0],
            [q_pos/8,  0,        q_vel/3, 0,       q_vel/2, 0,       0,     0],
            [0,        q_pos/8,  0,       q_vel/3, 0,       q_vel/2, 0,     0],
            [q_pos/6,  0,        q_vel/2, 0,       q_acc,   0,       0,     0],
            [0,        q_pos/6,  0,       q_vel/2, 0,       q_acc,   0,     0],
            [0,        0,        0,       0,       0,       0,       q_box, 0],
            [0,        0,        0,       0,       0,       0,       0,     q_box],
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

        self.B = np.array([  # control-input model
            [1, 0], 
            [0, 1], 
            [0, 0], 
            [0, 0], 
            [0, 0], 
            [0, 0], 
            [0, 0], 
            [0, 0]
        ])

        if cls == 0:
            r_pos = 15.0
            r_size = 5.0
        else:
            r_pos = 10.0
            r_size = 20.0
            
        self.R = np.diag([r_pos, r_pos, r_size, r_size])  # measurement noise covariance matrix

        self.track_age = 1  # number of total frames since initialization
        self.missed_frames = 0  # number of consecutive frames without update
        self.hits = 1  # number of successful updates
        self.is_confirmed = False  # track is confirmed after a few hits

    def predict(self, optical_flow):
        """
        Predict the next state and covariance of the Kalman filter.
        """
        # control vector
        u = np.array(optical_flow)  

        self.x = self.F @ self.x + self.B @ u  # predict state: x_k = F * x_{k-1} + B * u_k
        self.P = self.F @ self.P @ self.F.T + self.Q  # predict covariance

        self.track_age += 1  # increase track age
        self.missed_frames += 1  # increase missed frames count (no measurement update yet)

    def update(self, z):
        """
        Update the Kalman filter state with a new measurement.<br>
        Predict method has to be called beforehand.
        """

        y = z - (self.H @ self.x)  # innovation (measurement residual)
        S = self.H @ self.P @ self.H.T + self.R  # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Kalman gain

        self.x = self.x + K @ y  # update state estimate
        I = np.eye(8)  # identity matrix for covariance update
        self.P = (I - K @ self.H) @ self.P  # update covariance estimate

        self.missed_frames = 0  # reset missed frames

        if not self.is_confirmed:
            self.hits += 1  # increase hits (measurement updates)

    def mahalanobis(self, detections) -> list:
        """
        Returns Mahalanobis distance for a list of detections
        """
        S = self.H @ self.P @ self.H.T + self.R
        L = np.linalg.cholesky(S)
        y = np.array(detections) - (self.H @ self.x)
        z = np.linalg.solve(L, y.T).T
        return np.sqrt(np.sum(z**2, axis=1))

    @property  # decorator: to allow access like an attribute without calling as a method
    def box(self):
        """
        Returns the current bounding box estimate.
        """
        return np.array([self.x[0], self.x[1], self.x[6], self.x[7]])

    @property
    def velocity(self):
        """
        Returns the current velocity estimate.
        """
        return self.x[2:4]

class Tracker:
    def __init__(self):
        self.name = "Tracker"
        
        self.filters = []  # list of active Kalman filters (tracks)
        self.next_id = 1  # next unique track ID

        # separate birth/death for ball vs others
        self.ball_birth_threshold = 3  # frames needed to confirm a new ball track
        self.ball_death_threshold = 5  # frames without update before deleting a ball track
        self.birth_threshold = 10  # frames needed to confirm a new track
        self.death_threshold = 20  # frames without update before deleting a track
        self.output_threshold = 5  # if missing_frames is greater, dont return it as an active track, but dont delete it either
        self.assignment_threshold = 10.0  # maximum allowed cost for assigning detection to sa track 
        self.max_tracks = 25  # maximum number of active tracks

        # print("Module tracker started.")

    def start(self, data):
        """
        Required, but unused.
        """
        pass

    def stop(self, data):
        """
        Required, but unused
        """
        pass

    def iou(self, bt, bd):
        """
        Compute Intersection over Union (IoU) of two bounding boxes to determine similarity.
        """
        bt_x_left = bt[0] - bt[2] / 2
        bt_y_bottom = bt[1] - bt[3] / 2
        bt_x_right = bt[0] + bt[2] / 2
        bt_y_top = bt[1] + bt[3] / 2

        bd_x_left = bd[0] - bd[2] / 2
        bd_y_bottom = bd[1] - bd[3] / 2
        bd_x_right = bd[0] + bd[2] / 2
        bd_y_top = bd[1] + bd[3] / 2

        inter_x_left = max(bt_x_left, bd_x_left)
        inter_y_bottom = max(bt_y_bottom, bd_y_bottom)
        inter_x_right = min(bt_x_right, bd_x_right)
        inter_y_top = min(bt_y_top, bd_y_top)

        inter_area = max(0, inter_x_right - inter_x_left) * max(0, inter_y_top - inter_y_bottom)  # calculate intersection area

        bt_area = bt[2] * bt[3]
        bd_area = bd[2] * bd[3]
        union_area = bt_area + bd_area - inter_area

        if union_area == 0:
            return 0.0  # avoid division by zero
        return inter_area / union_area  # return IoU value between 0 and 1

    def _build_output(self):
        """
        Build output for all confirmed tracks.
        """
        confirmed = [f for f in self.filters if f.is_confirmed and f.missed_frames <= self.output_threshold]  # filter confirmed tracks

        # limit output to max_tracks oldest tracks if necessary
        if len(confirmed) > self.max_tracks:
            confirmed = sorted(confirmed, key=lambda f: f.track_age, reverse=True)[:self.max_tracks]  # sort by age and take oldest tracks

        boxes = [f.box.copy() for f in confirmed]
        velocities = [f.velocity.copy() for f in confirmed]
        ages = [f.track_age for f in confirmed]
        classes = [f.cls for f in confirmed]
        ids = [f.id for f in confirmed]

        if boxes:  # check if there are any confirmed tracks
            tracks = np.array(boxes)  # convert list of boxes to NumPy array
            trackVelocities = np.array(velocities)  # convert list of velocities to NumPy array
        else:  # no confirmed tracks
            tracks = np.zeros((0, 4), dtype=float)  # empty array for tracks
            trackVelocities = np.zeros((0, 2), dtype=float)  # empty array for velocities

        return {  # return output dictionary
            "tracks": tracks,
            "trackVelocities": trackVelocities,
            "trackAge": ages,
            "trackClasses": classes,
            "trackIds": ids
        }

    def step(self, data):
        """
        Main loop of the tracker.
        This method processes the input data, updates the filters, and returns the current state of the tracker.
        """
        detections = data.get("detections", [])  # get detections from input data
        detectionClasses = data.get("classes", [])  # get classes of detections
        opticalFlow = data.get("opticalFlow", (0,0))  # get optical flow

        if len(self.filters) == 0 and len(detections) > 0:  # if there are no filters...
            for det, cls in zip(detections, detectionClasses):  # ... create new filters for each detection
                f = Filter(det, cls)  # create a new filter
                f.hits = 1
                f.id = self.next_id
                self.next_id += 1
                self.filters.append(f)
            return self._build_output()

        if len(detections) == 0 and len(self.filters) > 0:  # if there are no detections...
            survivors = []  # ... keep existing filters that are still alive
            for f in self.filters:
                f.predict(opticalFlow)
                # class-specific thresholds
                if f.cls == 0:
                    birth_thr = self.ball_birth_threshold
                    death_thr = self.ball_death_threshold
                else:
                    birth_thr = self.birth_threshold
                    death_thr = self.death_threshold
                thresh = birth_thr if not f.is_confirmed else death_thr
                if f.missed_frames <= thresh:
                    survivors.append(f)  # only keep filters that are not dead
            self.filters = survivors
            return self._build_output()  # return current state of the tracker

        # predict states of all filters of current frame and build cost matrix
        cost_matrix = np.ones((len(self.filters), len(detections)), dtype=float)  # initialize cost matrix
        for i, f in enumerate(self.filters):
            f.predict(opticalFlow)
            mahal = f.mahalanobis(detections)
            for j, det in enumerate(detections):
                # euclidean gate
                if ((f.box[0] - det[0])**2 + (f.box[1] - det[1])**2)**0.5 < 15 and max([self.iou(fil.box, det) for fil in self.filters if fil is not f]) < 0.7:
                    cost_matrix[i, j] = (1.0 - self.iou(f.box, det)) * mahal[j]  # calculate IoU and fill cost matrix
                else:
                    cost_matrix[i, j] = 1000

        row_ind, col_ind = linear_sum_assignment(cost_matrix)  # solve assignment problem using Hungarian algorithm

        assigned_tracks = set()  # make a set of assigned tracks
        assigned_detections = set()  # make a set of assigned detections

        for r, c in zip(row_ind, col_ind):
            thresh = self.assignment_threshold if self.filters[r].cls != 0 else 100
            if cost_matrix[r, c] < thresh and self.filters[r].cls == detectionClasses[c]:  # IoU > iou_threshold as assignment threshold
                self.filters[r].update(detections[c])
                assigned_tracks.add(r)
                assigned_detections.add(c)
                if self.filters[r].id is None:
                    self.filters[r].id = self.next_id
                    self.next_id += 1

        for j, det in enumerate(detections):
            # if detection is non-ball but overlaps existing, skip only this one
            if detectionClasses[j] != 0 and max([self.iou(f.box, det) for f in self.filters]) > 0.5:
                continue
            if j not in assigned_detections:  # if detection was not assigned to a filter...
                new_filter = Filter(det, detectionClasses[j])  # create a new filter
                new_filter.hits = 1
                new_filter.id = self.next_id
                self.next_id += 1
                self.filters.append(new_filter)  # add new filter to the list

        survivors = []  # keep only filters that are still alive
        for f in self.filters:
            # class-specific thresholds
            if f.cls == 0:
                birth_thr = self.ball_birth_threshold
                death_thr = self.ball_death_threshold
            else:
                birth_thr = self.birth_threshold
                death_thr = self.death_threshold

            thresh = death_thr if f.is_confirmed else birth_thr  # set threshold based on confirmation status

            if f.missed_frames <= thresh:  # if filter is not dead...
                survivors.append(f)  # ... keep it in the list

            # confirm if minimum hits is reached
            if not f.is_confirmed and f.hits >= birth_thr:
                f.is_confirmed = True


        self.filters = survivors

        # return current state of the tracker
        return self._build_output()
