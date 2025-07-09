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


        boxes_with_scores = sorted(
            [
                (box, float(box.conf.item())) for box in results.boxes
            ],  # pairs box with confidence score
            key=lambda x: x[1],  # sorts by confidence
            reverse=True,  # highest confidence first
        )

        detections = []
        classes = []

        VALID_CLASSES = {
            0,
            1,
            2,
            3,
        }  # specifies relevant classes in case model changes in future

        for box, score in boxes_with_scores:
            # each box contains one detection result from YOLO (coordinates, class, confidence)

            cls_id = int(
                box.cls.item()
            )  # predicted class id (0 = ball, 1 = goalkeeper, 2 = player, 3 = referee)

            if cls_id not in VALID_CLASSES:
                continue  # skip other classes, if there are any

            # class specific thresholds
            if cls_id == 0:  # ball
                if score < 0.2:  # lower confidence, because detection is harder
                    continue
            else:  # player, goalkeeper, referee
                if score < 0.4:
                    continue

            x_center, y_center, width, height = box.xywh[
                0
            ].tolist()  # bounding box (center coordinates width, height)

            if cls_id == 0:  # shrinks bounding box of ball class by 30%
                width *= 0.7
                height *= 0.7

            detections.append(
                [x_center, y_center, width, height]
            )  # adds bounding box to list
            classes.append([cls_id])  # adds corresponding class id

        # converts valid detections to np arrays
        if detections:
            detections = np.array(detections).astype(
                np.float32
            )  # x_center, y_center, width, height
            classes = np.array(classes).astype(np.int32)  # class_id
        else:
            # returns empty arrays  if no detections are found
            detections = np.zeros((0, 4), dtype=np.float32)
            classes = np.zeros((0, 1), dtype=np.int32)

        return {"detections": detections, "classes": classes}
