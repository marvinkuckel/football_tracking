"""
Module detects objects (Ball, Goalkeeper, Player, Referee) per frame using a pretrained YOLOv8n model.

Outputs:
- detections: Bounding boxes (center x, center y, width, height)
- classes: Object classes (0 = Ball, 1 = Goalkeeper, 2 = Player, 3 = Referee)

Optimizations:
- Runs on GPU with half precision (FP16) if available
- Separate confidence thresholds for ball vs other objects

Limitations:
- Detection is frame-by-frame (no full tracking memory)
- Instability in class assignment
- Ball is sometimes not detected
- High input resolution (864) for better small object detection slows video down
"""

from ultralytics import YOLO
import torch
import numpy as np


class Detector:
    def __init__(self, weights="yolov8n-football.pt", imgsz=864, conf=0.2):
        """
        Directly loads the model on GPU, if available. 
        Otherwise falls back to CPU (much slower but still functional).
        """
        self.name = "Detector" 
        self.imgsz = imgsz  # higher resolution improves ball detection but decreases speed
        self.conf = conf  # confidence threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # checks if GPU is available
        self.model = YOLO(weights).to(self.device)  # loads weights
        if self.device == "cuda":
            self.model.half()  # converts model weights to half precision (FP16) on GPU for faster inference
        self.model.model.eval()  # optimizes for inference by turning off training behavior (dropouts, batch normalization)

        # warm up: runs dummy inferences to avoid first-frame lag
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)  # creates dummy black image (HxWxC)
        with torch.inference_mode():  # disables gradient tracking (since weights arent updated during inference)
            if self.device == "cuda":
                torch.cuda.synchronize()  
            for _ in range(5):  # runs 5 dummy inferences
                _ = self.model(dummy, imgsz=self.imgsz, conf=self.conf, verbose=False)
            if self.device == "cuda":
                torch.cuda.synchronize()  # ensures warm up is done before real inferences

        print(f"{self.name}: Model ready on {self.model.device}.")


    def start(self, data):
        # currently not needed
        print(f"{self.name}: Detector module started.")


    def stop(self, data):
        # currently not needed
        print(f"{self.name}: Detector module stopped.")


    def step(self, data):
        """
        Runs YOLO detection on the current frame and returns bounding boxes and class IDs.
        Called once per frame.
        """
        image = data["image"]  # gets current frame from input dict

        # fast inference in "no-training mode" (no gradient tracking)
        with torch.inference_mode():
            # calls the model directly (slightly faster)
            results = self.model(
                image,
                imgsz=self.imgsz,  # image size specified in __init__
                conf=self.conf,    # confidence threshold specified in __init__
                verbose=False      
            )[0]  # returns list of results (one per image), since we call per frame we only take the first

        # converts detection tensors to numpy arrays
        boxes   = results.boxes.xywh.cpu().numpy()  # (N, 4): coordinates & size for each box (x_center, y_center, width, height)
        classes = results.boxes.cls.cpu().numpy().astype(int)  # (N,): class ids as int (0=ball, 1=goalkeeper, 2=player, 3=referee)
        confs   = results.boxes.conf.cpu().numpy()  # (N,): confidence for each box as float (between 0.0 and 1.0)

        # bool mask for valid detections
        valid_classes = (classes >= 0) & (classes <= 3)  # only keeps relevant classes (0-3)
        # bool mask for class specific confidence thresholds
        ball_mask = (classes == 0) & (confs >= 0.2)  # lower confidence for ball, because detection is harder
        other_mask = (classes != 0) & (confs >= 0.4)  # higher confidence for player, goalkeeper, referee

        # combines everything into a single mask
        mask = valid_classes & (ball_mask | other_mask)

        filtered_boxes   = boxes[mask]
        filtered_classes = classes[mask]
        
        # vectorized mask for shrinking ball bounding boxes
        ball_idx = (filtered_classes == 0)  # bool mask for class ball
        filtered_boxes[ball_idx, 2:] *= 0.7  # shrink width & height by 30%

        # prepares outputs in required shapes
        detections = filtered_boxes.astype(np.float32)
        det_classes = filtered_classes.reshape(-1, 1).astype(np.int32)  # already handles correct output in case of no detections


        return {"detections": detections, "classes": det_classes}

