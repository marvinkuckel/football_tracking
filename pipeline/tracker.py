import numpy as np


class Filter:
    """
    Tracks a single object for multiple frames.
    Stores its bounding box, class, velocity, age, missed frames, and pending-birth state.
    """
    def __init__(self, z, cls, use_markow=False):
        self.box = z.copy()             # bounding box [x,y,w,h] with x,y = center coordinates, w,h = width & height
        self.cls = cls                  # detected class (0=ball, 1=goalkeeper, 2=player, 3=referee)
        self.velocity = np.zeros(2)     # velocity (dx, dy) vector
        self.track_age = 1              # how many frames the object was tracked
        self.missed_frames = 0          # how many frames the object couldn't be matched to a detection 
        self.id = None                  # unique id for each track

        # tracks only get birthed when they are matched x consecutive times 
        self.hits = 0                   # how many consecutive matches have been found
        self.is_confirmed = False       # becomes True once hits ≥ birth_threshold
        
        #markow 
        self.use_markow = use_markow
        self.markow_noise_std = 3.0      # standard deviation for markow noise in pixels

    def predict(self, optical_flow):
        """
        Called when a track isn't matched to a detection.
        Predicts its new box by shifting it by the objects velocity and optical flow.
        """
        cam_dx, cam_dy = optical_flow                   # camera movement
        
        if self.use_markow:
            # stochastic prediction using markow model
            predicted_box = self.markov_predict_position()
            predicted_box[0] += cam_dx
            predicted_box[1] += cam_dy
            self.box = predicted_box
            
        else:
            # Predicts new position by adding velocity and camera movement
            self.box[0] += self.velocity[0] + cam_dx        # shifts x coordinate 
            self.box[1] += self.velocity[1] + cam_dy        # shifts y coordinate
            
        self.track_age += 1                             # increments age by one frame
        self.missed_frames += 1                         # increments the counter for missed detection match

    def update(self, z, optical_flow):
        """
        Called when a track is matched to a detection.
        Resets its box, updates its speed, and clears missed_frames.
        """
        # calculates displacement of the track
        disp_x = z[0] - self.box[0]
        disp_y = z[1] - self.box[1]
        
        # subtracts camera movement from displacement
        cam_dx, cam_dy = optical_flow
        obj_vx = disp_x - cam_dx
        obj_vy = disp_y - cam_dy
        
        # stores tracks velocity
        self.velocity = np.array([obj_vx, obj_vy])
        
        # updates box & ages
        self.box = z.copy()         # takes over bounding box of new matched detection
        self.track_age += 1         # increments age by one frame
        self.missed_frames = 0      # resets the missed age, since the track was matched to a detection
        
    def markov_predict_position(self):
        """
        Returns a stochastic predicted position that is extended by a random component.
        """
        noise = np.random.normal(0, self.markov_noise_std, 2)
        predicted_center = self.box[:2] + self.velocity + noise
        return np.array([
            predicted_center[0],
            predicted_center[1],
            self.box[2],
            self.box[3]
        ])



class Tracker:
    """
    Multi‐target tracker with simple birth/death logic and matching through IoU and hungarian data association.
    Manages a list of Filter objects (tracks) and attempts to match incoming detections each frame.
    Updates existing tracks, prunes stale ones, and spawns new pending tracks.
    """
    def __init__(self):
        """
        Constructs Tracker object.
        Note: Most initialization happens in start().
        """
        self.name = "Tracker"       # Do not change the name of the module as otherwise recording replay would break!

    def start(self, data):
        """
        Initializes the tracker with empty filter list, id counter and thresholds.
        data is currently unused.
        """
        self.filters = []           # confirmed and pending filters
        self.next_id = 1            # unique id counter

        self.birth_threshold = 3    # needs n consecutive matches before track confirmation
        self.death_threshold = 8    # prune after 8 misses
        self.max_tracks = 25        # caps output tracks

        print("Module tracker started.")

    def stop(self, data):
        """
        Currently, no special shutdown procedure is needed.
        data is currently unused.
        """
        print("Module tracker stopped.")

    def iou(self, bt, bd):
        """
        Computes the intersection over union, so the overlap, of a tracks and a detections bounding box for later matching.
        bt, bd: [x_center, y_center, width, height]
        Returns 0.0 <= IoU <= 1.0
        """
        # edges of tracks bounding box
        bt_x_left = bt[0] - bt[2] / 2       # left edge x
        bt_y_bottom = bt[1] - bt[3] / 2     # bottom edge y
        bt_x_right = bt[0] + bt[2] / 2      # right edge x
        bt_y_top = bt[1] + bt[3] / 2        # top edge y
    
        # edges of detections bounding box
        bd_x_left = bd[0] - bd[2] / 2       # left edge x
        bd_y_bottom = bd[1] - bd[3] / 2     # bottom edge y
        bd_x_right = bd[0] + bd[2] / 2      # right edge x
        bd_y_top = bd[1] + bd[3] / 2        # top edge y

        # edges of intersection rectangle
        inter_x_left = max(bt_x_left, bd_x_left)                # left edge x
        inter_y_bottom = max(bt_y_bottom, bd_y_bottom)          # bottom edge y
        inter_x_right = min(bt_x_right, bd_x_right)             # right edge x
        inter_y_top = min(bt_y_top, bd_y_top)                   # top edge y

        # intersection rectangle area
        inter_area = max(0, inter_x_right - inter_x_left) * max(0, inter_y_top - inter_y_bottom)    # clamps at 0 if there is no intersection

        # union area
        bt_area = bt[2] * bt[3]                         # area of tracks bounding box
        bd_area = bd[2] * bd[3]                         # area of detections bounding box
        union_area = bt_area + bd_area - inter_area

        # intersection over union
        if union_area == 0:                             # avoids 0 division error if union == 0
            return 0.0
        else:
            iou = inter_area / union_area
            return iou
        
    def _build_output(self):
        """   
        Collects all confirmed tracks, caps them by age.
        Returns required dict and ensures proper shapes:
        numpy arrays for boxes and velocities, lists for ages, classes, and ids.
        """
        # select only confirmed tracks
        confirmed = [f for f in self.filters if f.is_confirmed]

        # cap to the oldest self.max_tracks
        if len(confirmed) > self.max_tracks:
            confirmed = sorted(                                     
                confirmed, key=lambda f: f.track_age, reverse=True  # sort by track_age descending
            )[: self.max_tracks]                                    # slice after max_tracks

        # build lists
        boxes = [f.box.copy() for f in confirmed]
        velocities = [f.velocity.copy() for f in confirmed]
        ages = [f.track_age for f in confirmed]  
        classes = [f.cls for f in confirmed]  
        ids = [f.id for f in confirmed]  

        # ensures 2D array is returned, even if no confirmed tracks exist
        if boxes:
            tracks = np.array(boxes)                    
            trackVelocities = np.array(velocities)      
        else:
            tracks          = np.zeros((0, 4), dtype=float)     
            trackVelocities = np.zeros((0, 2), dtype=float)

        # return dictionary
        return {
            "tracks": tracks,                       # Nx4 array
            "trackVelocities": trackVelocities,     # Nx2 array
            "trackAge": ages,                       # Nx1 list
            "trackClasses": classes,                # Nx1 list
            "trackIds": ids                         # Nx1 list
        }

    def step(self, data):
        # for finding best matches with hungarian algorithm
        from scipy.optimize import linear_sum_assignment

        detections = data["detections"]         # Nx4 array: x_center, y_center, width, height
        detectionClasses = data["classes"]      # Nx1 array: 0=Ball, 1=GoalKeeper, 2=Player, 3=Referee
        opticalFlow = data["opticalFlow"]       # 1x2 array: x_shift, y_shift

        # case 1: no tracks, but detections exist
        if len(self.filters) == 0 and len(data["detections"]) > 0:
            # create new filter objects (pending tracks) for each detection
            for det, cls in zip(detections, detectionClasses):
                f = Filter(det, cls, use_markov=True)  # use_markow=True for stochastic prediction
                f.hits = 1
                f.id = self.next_id
                self.next_id += 1
                self.filters.append(f)
            return self._build_output()
        
        # case 2: no detections, but tracks exist
        if len(data["detections"]) == 0 and len(self.filters) > 0:
            # predict tracks new position and prunes tracks over the max threshold for missing age
            survivors = []
            for f in self.filters:
                f.predict(opticalFlow)
                # use birth_threshold for un-confirmed (to prune stale tracks earlier), death_threshold for confirmed
                thresh = self.birth_threshold if not f.is_confirmed else self.death_threshold
                if f.missed_frames <= thresh:
                    survivors.append(f)
            self.filters = survivors
            return self._build_output()
        
        # case 3: tracks and detections exist
        nt = len(self.filters)   # number of current tracks (pending and confirmed)
        nd = len(detections)     # number of current detections

        # cost matrix step 1: create numpy array in the shape of nt * nd
        cost_matrix = np.zeros((nt, nd), dtype=float)

        # cost matrix step 2: fill numpy array with calculated costs for track-detection pairs
        for i, track in enumerate(self.filters):
            for j, detection in enumerate(detections):
                cost_matrix[i, j] = 1.0 - self.iou(track.box, detection)
                    
        # hungarian algorithm to get the best global matches (returns indieces)
        track_indices, detection_indices = linear_sum_assignment(cost_matrix)

        # decide on valid matches with IoU treshold
        min_iou = 0.2
        matches = []
        all_track_ids = set(range(len(self.filters)))
        all_det_ids = set(range(len(detections)))
        matched_tracks = set()
        matched_detections = set()

        for ti, di in zip(track_indices, detection_indices):
            # cost = 1 − iou
            iou_val = 1.0 - cost_matrix[ti, di]
            if iou_val >= min_iou:
                matches.append((ti, di))
                matched_tracks.add(ti)
                matched_detections.add(di)
            # otherwise treated as unmatched

        # call update() for track at index ti with matched detection at index di
        for ti, di in matches:
            f = self.filters[ti]
            f.update(detections[di], opticalFlow)       
            # delayed-birth logic
            if not f.is_confirmed:
                f.hits += 1                             # adds one match to count
                if f.hits >= self.birth_threshold:
                    f.is_confirmed = True               # if match treshold is reached, track is born
    
        # build sets of unmatched tracks and detections
        unmatched_tracks     = all_track_ids - matched_tracks       
        unmatched_detections = all_det_ids - matched_detections

        # call predict() for unmatched tracks
        # remove in descending order so earlier deletes don't shift later indices
        for ti in sorted(unmatched_tracks, reverse=True):
            f = self.filters[ti]
            f.predict(opticalFlow)
            # use birth_threshold for un-confirmed (to prune stale tracks earlier), death_threshold for confirmed
            thresh = self.birth_threshold if not f.is_confirmed else self.death_threshold
            if f.missed_frames > thresh:
                del self.filters[ti]                    # if threshold is exceeded, track is pruned

        # create pending tracks for unmatched detections
        for di in sorted(unmatched_detections):
            det = detections[di]
            cls = detectionClasses[di]
            f = Filter(det, cls)
            f.hits = 1
            f.id = self.next_id
            self.next_id += 1
            self.filters.append(f)
        
        return self._build_output()
