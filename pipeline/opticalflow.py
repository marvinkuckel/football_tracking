import numpy as np
import cv2

class OpticalFlow:
    def __init__(self):
        self.name = "Optical Flow" # Do not change the name of the module as otherwise recording replay would break!
        self.pre_image = None
        self.pre_points = None

    def start(self, data):
        self.pre_image = None
        self.pre_points = None

    def stop(self, data):
        self.pre_image = None
        self.pre_points = None

    def step(self, data):
        # TODO: Implement processing of a single frame
        frame = data["image"]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.pre_image is None:
            self.pre_image = gray
            return {
                "opticalFlow": np.array([0.0, 0.0], dtype = np.float32)
            }

        if self.pre_points is None:
            self.pre_points = cv2.goodFeaturesToTrack(
                self.pre_image,  # Shi-Tomasi
                maxCorners = 100,
                qualityLevel = 0.3,
                minDistance = 7,
                blockSize = 7
            )
        
        next_points, status, _ = cv2.calcOpticalFlowPyrLK(
            self.pre_image, gray, self.pre_points, None
            )
        
        good_old = self.pre_points[status == 1]  # (1 = True)
        good_new = next_points[status == 1]

        if len(good_old) > 0:  # If there are points to track...
            flow = good_new - good_old  # vector flow
            mean_flow = np.mean(flow, axis=0)
        else:
            mean_flow = np.array([0.0, 0.0])

        self.pre_image = gray

        self.pre_points = good_new.reshape(-1, 1, 2) if len(good_new) > 0 else None  # Reset

        for i, (new, old) in enumerate(zip(good_new, good_old)):  # Unpack the points into x, y coordinates
            a, b = new.ravel()
            c, d = old.ravel()
            cv2.arrowedLine(frame, (int(c), int(d)), (int(a), int(b)), (0, 255, 0), 2, tipLength = 0.05)  # Arrowed line

        cv2.waitKey(1)

        # The task of the optical flow module is to determine the overall avergae pixel shift between this and the previous image. 
        # You 

        # Note: You can access data["image"] to receive the current image
        # Return a dictionary with the motion vector between this and the last frame
        #
        # The "opticalFlow" signal must contain a 1x2 NumPy Array with the X and Y shift (delta values in pixels) of the image motion vector
        return {
           "opticalFlow": mean_flow.astype(np.float32)
        }

