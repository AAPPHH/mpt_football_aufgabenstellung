import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

class Tracker:
    """
    A tracker class that utilizes the DeepSort algorithm to track objects
    across sequential frames. It maintains internal state for ages, classes,
    previous positions, and velocities for each track.
    """
    
    def __init__(
        self,
        max_iou_distance=0.7,
        max_age=30,
        n_init=3,
        nms_max_overlap=1.0,
        max_cosine_distance=0.2,
        embedder="mobilenet",
        embedder_gpu=True,
        half=False,
        bgr=True,
    ):
        """
        Initialize the DeepSort tracker and relevant parameters for tracking.

        Args:
            max_iou_distance (float): Maximum IoU distance for matching.
            max_age (int): Maximum number of missed updates for a track before it is deleted.
            n_init (int): Number of consecutive detections before a track is confirmed.
            nms_max_overlap (float): Non-Maximum Suppression threshold.
            max_cosine_distance (float): Maximum cosine distance for appearance matching.
            embedder (str): The name of the embedder model used for feature extraction.
            embedder_gpu (bool): Whether to use the GPU for the embedder.
            half (bool): If True, use FP16 precision where supported.
            bgr (bool): If True, interpret images as BGR (OpenCV format); otherwise RGB.
        """
        self.name = "Tracker"
        self.deepsort = DeepSort(
            max_iou_distance=max_iou_distance,
            max_age=max_age,
            n_init=n_init,
            nms_max_overlap=nms_max_overlap,
            max_cosine_distance=max_cosine_distance,
            embedder=embedder,
            embedder_gpu=embedder_gpu,
            half=half,
            bgr=bgr,
        )

        self.track_ages = {}
        self.track_classes = {}
        self.prev_positions = {}
        self.track_velocities = {}

    def start(self, data):
        """
        Start the DeepSort tracker.
        
        Args:
            data (dict): A dictionary potentially containing initialization data.
        """
        print("[INFO] DeepSort Tracker wurde gestartet.")

    def stop(self, data):
        """
        Stop the DeepSort tracker.
        
        Args:
            data (dict): A dictionary potentially containing data for stopping the tracker.
        """
        print("[INFO] DeepSort Tracker wurde gestoppt.")

    def _xywh_to_xyxy(self, x_center, y_center, w, h):
        """
        Convert bounding box format from (x_center, y_center, w, h) to (x1, y1, x2, y2).

        Args:
            x_center (float): The x-coordinate of the center of the bounding box.
            y_center (float): The y-coordinate of the center of the bounding box.
            w (float): The width of the bounding box.
            h (float): The height of the bounding box.

        Returns:
            list: A list [x1, y1, x2, y2] representing the bounding box.
        """
        x1 = x_center - w / 2
        y1 = y_center - h / 2
        x2 = x_center + w / 2
        y2 = y_center + h / 2
        return [x1, y1, x2, y2]

    def _xyxy_to_xywh(self, x1, y1, x2, y2):
        """
        Convert bounding box format from (x1, y1, x2, y2) to (x_center, y_center, w, h).

        Args:
            x1 (float): The left coordinate of the bounding box.
            y1 (float): The top coordinate of the bounding box.
            x2 (float): The right coordinate of the bounding box.
            y2 (float): The bottom coordinate of the bounding box.

        Returns:
            list: A list [x_center, y_center, w, h] representing the bounding box.
        """
        w = x2 - x1
        h = y2 - y1
        x_center = x1 + w / 2
        y_center = y1 + h / 2
        return [x_center, y_center, w, h]

    def compute_iou(self, boxA, boxB):
        """
        Compute the Intersection-over-Union (IoU) for two bounding boxes in (x1, y1, x2, y2) format.

        Args:
            boxA (list): The first bounding box [x1, y1, x2, y2].
            boxB (list): The second bounding box [x1, y1, x2, y2].

        Returns:
            float: The IoU value between 0.0 and 1.0.
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interW = max(0, xB - xA)
        interH = max(0, yB - yA)
        interArea = interW * interH
        if interArea == 0:
            return 0.0
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def step(self, data):
        """
        Process a single frame of detections and update tracking information.

        This method uses the DeepSort tracker to update the states of tracked objects
        based on current frame detections. It calculates velocities, ages, and class
        information for each tracked object.

        Args:
            data (dict): A dictionary containing:
                - 'detections' (list): List of detections in (x_center, y_center, w, h) format.
                - 'classes' (list): List of class IDs corresponding to each detection.
                - 'image' (ndarray): The current frame (image) in numpy array format.
                - 'teamClasses' (list, optional): Additional class information for teams.

        Returns:
            dict: A dictionary with tracking results containing:
                - "tracks": Nx4 array of current positions in (x_center, y_center, w, h) format.
                - "trackVelocities": Nx2 array of velocity vectors for each track.
                - "trackAge": List of ages corresponding to each track.
                - "trackClasses": List of class IDs for each tracked object.
                - "trackIds": List of assigned track IDs.
                - "teamClasses": List of team class IDs (if provided).
        """
        detections = data.get("detections", [])
        classes = data.get("classes", [])
        frame = data.get("image", None)

        if frame is None:
            print("[WARN] Kein Frame übergeben, ReID nicht möglich!")
            return {
                "tracks": np.zeros((0, 4)),
                "trackVelocities": np.zeros((0, 2)),
                "trackAge": [],
                "trackClasses": [],
                "trackIds": [],
                "teamClasses": [],
            }

        if len(detections) == 0:
            return {
                "tracks": np.zeros((0, 4)),
                "trackVelocities": np.zeros((0, 2)),
                "trackAge": [],
                "trackClasses": [],
                "trackIds": [],
                "teamClasses": [],
            }

        raw_detections = []
        detection_boxes = []
        detection_classes = []
        for i, det in enumerate(detections):
            x_center, y_center, w, h = det
            x1, y1, x2, y2 = self._xywh_to_xyxy(x_center, y_center, w, h)
            left, top = x1, y1
            ww = x2 - x1
            hh = y2 - y1
            conf = 1.0
            cls_val = int(classes[i]) if i < len(classes) else 0
            raw_detections.append(([left, top, ww, hh], conf, cls_val))
            detection_boxes.append([x1, y1, x2, y2])
            detection_classes.append(cls_val)

        outputs = self.deepsort.update_tracks(raw_detections, frame=frame)

        tracked_positions = []
        tracked_velocities = []
        tracked_ages = []
        tracked_classes = []
        tracked_ids = []

        for t in outputs:
            if not t.is_confirmed() or t.time_since_update > 0:
                continue

            track_id = t.track_id
            l, t_, r, b = t.to_tlbr()
            x_center, y_center, w_, h_ = self._xyxy_to_xywh(l, t_, r, b)

            if track_id not in self.track_ages:
                self.track_ages[track_id] = 1
            else:
                self.track_ages[track_id] += 1

            best_iou = 0.0
            best_cls = 0
            track_box = [l, t_, r, b]
            for i, det_box in enumerate(detection_boxes):
                iou = self.compute_iou(track_box, det_box)
                if iou > best_iou:
                    best_iou = iou
                    best_cls = detection_classes[i]
            if best_iou > 0.3:
                cls_in_tracker = best_cls
            else:
                cls_in_tracker = 0

            self.track_classes[track_id] = cls_in_tracker

            if track_id in self.prev_positions:
                px_center, py_center, _, _ = self.prev_positions[track_id]
                vx = x_center - px_center
                vy = y_center - py_center
                self.track_velocities[track_id] = (vx, vy)
            else:
                self.track_velocities[track_id] = (0.0, 0.0)

            self.prev_positions[track_id] = (x_center, y_center, w_, h_)

            tracked_positions.append([x_center, y_center, w_, h_])
            tracked_velocities.append(self.track_velocities[track_id])
            tracked_ages.append(self.track_ages[track_id])
            tracked_classes.append(cls_in_tracker)
            tracked_ids.append(track_id)

        team_classes = data.get("teamClasses", [])
        if len(team_classes) == 0:
            team_classes = [0] * len(tracked_positions)

        if len(tracked_positions) == 0:
            tracks = np.zeros((0, 4))
        else:
            tracks = np.array(tracked_positions).reshape(-1, 4)

        if len(tracked_velocities) == 0:
            track_velocities = np.zeros((0, 2))
        else:
            track_velocities = np.array(tracked_velocities).reshape(-1, 2)

        results = {
            "tracks": tracks,
            "trackVelocities": track_velocities,
            "trackAge": tracked_ages,
            "trackClasses": tracked_classes,
            "trackIds": tracked_ids,
            "teamClasses": team_classes,
        }

        return results