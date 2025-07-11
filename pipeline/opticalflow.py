import numpy as np
import cv2


class OpticalFlow:
    def __init__(self):
        self.name = "Optical Flow"  # Do not change the name of the module as otherwise recording replay would break!
        self.pre_image = None
        self.pre_points = None

    def start(self, data):
        """
        Initialize/reset previous image and points on start
        """
        self.pre_image = None
        self.pre_points = None

    def stop(self, data):
        """
        Cleanup on stopping the module: release stored data
        """
        self.pre_image = None
        self.pre_points = None

    def step(self, data):
        """
        Process a single frame/image
        """
        frame = data["image"]  # current image frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert to grayscale

        if self.pre_image is None:  # if no previous image exists...
            self.pre_image = gray  # ... set the current image as reference
            return {"opticalFlow": np.array([0.0, 0.0], dtype=np.float32)}  # return zero flow if no previous image exists

        if self.pre_points is None:  # if no previous points exist...
            self.pre_points = cv2.goodFeaturesToTrack(  # detect corners (good features) in the previous image, Shi-Tomasi
                self.pre_image,  # previous image
                maxCorners=100,  # maximum number of corners to return/points to find
                qualityLevel=0.3,  # minimal acceptable quality of corners
                minDistance=7,  # minimal distance between corners
                blockSize=7,  # size of the block for computing covariance matrix
            )

        next_points, status, _ = cv2.calcOpticalFlowPyrLK(  # Lukas-Kanade
            self.pre_image, gray, self.pre_points, None
        )

        good_old = self.pre_points[status == 1]  # (1 = True)
        good_new = next_points[status == 1]

        if len(good_old) > 0:  # If there are points to track...
            flow = good_new - good_old  # vector flow
            mean_flow = np.mean(flow, axis=0)  # average flow vector
        else:  # If no points are tracked...
            mean_flow = np.array([0.0, 0.0])  # ... set flow to zero

        self.pre_image = gray  # update the previous image to the current one

        self.pre_points = (
            good_new.reshape(-1, 1, 2) if len(good_new) > 0 else None  # reshape for OpenCV
        )  # Reset

        for i, (new, old) in enumerate(
            zip(good_new, good_old)
        ):  # Unpack the points into x, y coordinates
            a, b = new.ravel()
            c, d = old.ravel()
            cv2.arrowedLine(
                frame,
                (int(c), int(d)),
                (int(a), int(b)),
                (0, 255, 0),
                2,
                tipLength=0.05,
            )  # Arrowed line

        # The task of the optical flow module is to determine the overall avergae pixel shift between this and the previous image.
        # You

        # Note: You can access data["image"] to receive the current image
        # Return a dictionary with the motion vector between this and the last frame
        #
        # The "opticalFlow" signal must contain a 1x2 NumPy Array with the X and Y shift (delta values in pixels) of the image motion vector
        return {"opticalFlow": mean_flow.astype(np.float32)}
