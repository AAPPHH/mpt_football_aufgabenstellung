from ultralytics import YOLO
import numpy as np

class Detector:
    """
    A class that uses YOLOv8 for detecting specific objects (e.g., players, ball, etc.)
    in an input image. The model path and confidence threshold can be configured.
    """

    def __init__(self, model_path="yolov8m-football.pt", conf_threshold=0.1):
        """
        Initialize the Detector.

        Args:
            model_path (str): Path to the YOLOv8 model weights file.
            conf_threshold (float): Confidence threshold for filtering out low-confidence detections.
        """
        self.name = "Detector"
        self.model_path = model_path
        self.conf_threshold = conf_threshold

    def start(self, data):
        """
        Load the YOLOv8 model when the detector starts.

        Args:
            data (dict): A dictionary containing initialization parameters (if any).
        """
        print("[INFO] YOLOv8 Detector started.")
        self.model = YOLO(self.model_path)

    def stop(self, data):
        """
        Perform any necessary cleanup when the detector stops.

        Args:
            data (dict): A dictionary containing parameters for stopping (if any).
        """
        print("[INFO] YOLOv8 Detector stopped.")

    def step(self, data):
        """
        Perform object detection on the current frame/image using the YOLOv8 model.

        Args:
            data (dict): Must contain the key 'image' associated with the input image (NumPy array).

        Returns:
            dict: A dictionary containing:
                - "detections": N x 4 array of detections in [x_center, y_center, width, height] format.
                - "classes": An array of class IDs corresponding to the detected objects.
        """
        image = data["image"]
        results = self.model.predict(image, verbose=False)
        boxes = results[0].boxes

        detections = []
        classes = []

        for box in boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            if conf < self.conf_threshold:
                continue

            # Keep only certain classes (e.g. 0,1,2,3)
            if cls_id in [0, 1, 2, 3]:
                xywh = box.xywh[0].cpu().numpy()
                detections.append(xywh)
                classes.append(cls_id)
            else:
                continue

        # Convert lists to numpy arrays, or return empty arrays if no detections
        if detections:
            detections = np.stack(detections)
            classes = np.array(classes).reshape(-1)
        else:
            detections = np.zeros((0, 4))
            classes = np.zeros((0,), dtype=int)

        return {"detections": detections, "classes": classes}