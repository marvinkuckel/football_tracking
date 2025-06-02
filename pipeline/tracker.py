import numpy as np


class Filter:
    """
    Tracks a single object for multiple frames.
    Stores its bounding box, class, velocity, age, missed frames, and pending-birth state.
    """
    def __init__(self, z, cls):
        self.box = z.copy()             # bounding box [x,y,w,h] with x,y = center coordinates, w,h = width & height
        self.cls = cls                  # detected class (0=ball, 1=goalkeeper, 2=player, 3=referee)
        self.velocity = np.zeros(2)     # velocity (dx, dy) vector
        self.track_age = 1              # how many frames the object was tracked
        self.missed_frames = 0          # how many frames the object couldn't be matched to a detection 
        self.id = None                  # unique id for each track

        # tracks only get birthed when they are matched x consecutive times 
        self.hits = 0                   # how many consecutive matches have been found
        self.is_confirmed = False       # becomes True once hits ≥ birth_threshold

    def predict(self, optical_flow):
        """
        Called when a track isn't matched to a detection.
        Predicts its new box by shifting it by the objects velocity and optical flow.
        """
        cam_dx, cam_dy = optical_flow                   # camera movement
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
