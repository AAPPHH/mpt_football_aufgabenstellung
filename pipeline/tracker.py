import numpy as np
from scipy.optimize import linear_sum_assignment


class Filter:
    """
    Kalman filter for tracking a single object in 2D bounding box coordinates.
    """

    _next_id = 1

    def __init__(self, z, cls, Q=None, R=None):
        """
        Initializes the filter with an initial measurement.

        Args:
            z (array-like): Initial measurement [x, y, w, h].
            cls (int): Class label.
            Q (ndarray, optional): Process noise covariance (6x6).
            R (ndarray, optional): Measurement noise covariance (4x4).
        """
        z = np.asarray(z, dtype=np.float32)
        self.x = np.zeros((6,), dtype=np.float32)
        self.x[:4] = z
        self.x[4:] = 0.0

        P_pos = 10.0
        P_vel = 1000.0
        self.P = np.diag([P_pos, P_pos, P_pos, P_pos, P_vel, P_vel]).astype(np.float32)

        self.F = np.eye(6, dtype=np.float32)
        dt = 1.0
        self.F[0, 4] = dt
        self.F[1, 5] = dt

        self.H = np.zeros((4, 6), dtype=np.float32)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0

        self.Q = Q if Q is not None else np.eye(6, dtype=np.float32)
        self.R = R if R is not None else np.eye(4, dtype=np.float32)

        self.id = Filter._next_id
        Filter._next_id += 1

        self.cls = cls
        self.age = 1
        self.misses = 0

    def predict(self):
        """Predicts the state ahead one time step."""
        self.x = self.F.dot(self.x)
        self.P = self.F.dot(self.P).dot(self.F.T) + self.Q

    def update(self, z):
        """
        Updates the state with a new measurement.

        Args:
            z (array-like): Measurement [x, y, w, h].
        """
        z = np.asarray(z, dtype=np.float32)
        y = z - self.H.dot(self.x)
        S = self.H.dot(self.P).dot(self.H.T) + self.R
        K = self.P.dot(self.H.T).dot(np.linalg.inv(S))
        self.x = self.x + K.dot(y)
        I = np.eye(6, dtype=np.float32)
        self.P = (I - K.dot(self.H)).dot(self.P)
        self.misses = 0

    def increment_age(self):
        """Increments the track age."""
        self.age += 1

    def mark_missed(self):
        """Marks the track as missed (no update)."""
        self.misses += 1

    def is_deleted(self, max_misses):
        """
        Checks if track should be deleted based on misses.

        Args:
            max_misses (int)

        Returns:
            bool: True if misses >= max_misses.
        """
        return self.misses >= max_misses

    def get_state(self):
        """
        Returns current bounding box.

        Returns:
            ndarray: [x, y, w, h].
        """
        return self.x[:4].copy()

    def get_velocity(self):
        """
        Returns current velocity estimate.

        Returns:
            ndarray: [vx, vy].
        """
        return self.x[4:].copy()


class Tracker:
    """
    Multi-object tracker using Kalman filters and Hungarian assignment.
    """

    def __init__(self):
        """Initializes the tracker."""
        self.name = "Tracker"
        self.filters = []
        self.max_misses = 5
        self.iou_threshold = 0.3

    def start(self, data):
        """
        Starts the tracker, clearing existing tracks.

        Args:
            data: Ignored.
        """
        self.filters = []
        Filter._next_id = 1

    def stop(self, data):
        """
        Stops the tracker, clearing all tracks.

        Args:
            data: Ignored.
        """
        self.filters = []

    def step(self, data):
        """
        Processes a frame of detections.

        Args:
            data (dict): Contains 'detections' (Nx4) and 'classes' (N).

        Returns:
            dict: 'tracks', 'trackVelocities', 'trackAge', 'trackClasses', 'trackIds'.
        """
        dets = np.asarray(data.get("detections", []), dtype=np.float32)
        classes = list(data.get("classes", []))

        for f in self.filters:
            f.predict()

        T = len(self.filters)
        D = dets.shape[0]
        matches = []
        unmatched_tracks = list(range(T))
        unmatched_dets = list(range(D))

        if T > 0 and D > 0:
            cost_matrix = np.zeros((T, D), dtype=np.float32)
            for i, f in enumerate(self.filters):
                tb = f.get_state()
                for j in range(D):
                    cost_matrix[i, j] = 1.0 - self._iou(tb, dets[j])
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            unmatched_tracks = list(range(T))
            unmatched_dets = list(range(D))
            matches = []
            for r, c in zip(row_ind, col_ind):
                if cost_matrix[r, c] <= 1.0 - self.iou_threshold:
                    matches.append((r, c))
                    unmatched_tracks.remove(r)
                    unmatched_dets.remove(c)

        for r, c in matches:
            f = self.filters[r]
            f.update(dets[c])
            f.cls = classes[c]
            f.increment_age()

        for idx in unmatched_tracks:
            f = self.filters[idx]
            f.mark_missed()
            f.increment_age()

        for j in unmatched_dets:
            nf = Filter(dets[j], classes[j])
            self.filters.append(nf)

        self.filters = [f for f in self.filters if not f.is_deleted(self.max_misses)]

        if self.filters:
            tracks = np.stack([f.get_state() for f in self.filters]).astype(np.float32)
            trackVelocities = np.stack([f.get_velocity() for f in self.filters]).astype(np.float32)
        else:
            tracks = np.empty((0, 4), dtype=np.float32)
            trackVelocities = np.empty((0, 2), dtype=np.float32)

        trackAge = [f.age for f in self.filters]
        trackClasses = [f.cls for f in self.filters]
        trackIds = [f.id for f in self.filters]

        return {
            "tracks": tracks,
            "trackVelocities": trackVelocities,
            "trackAge": trackAge,
            "trackClasses": trackClasses,
            "trackIds": trackIds,
        }

    @staticmethod
    def _iou(b1, b2):
        """
        Computes IoU for two bounding boxes in [x, y, w, h].

        Args:
            b1, b2 (array-like): Bounding boxes.
        Returns:
            float: IoU value.
        """
        x11 = b1[0] - b1[2] / 2
        y11 = b1[1] - b1[3] / 2
        x12 = b1[0] + b1[2] / 2
        y12 = b1[1] + b1[3] / 2

        x21 = b2[0] - b2[2] / 2
        y21 = b2[1] - b2[3] / 2
        x22 = b2[0] + b2[2] / 2
        y22 = b2[1] + b2[3] / 2

        xi1 = max(x11, x21)
        yi1 = max(y11, y21)
        xi2 = min(x12, x22)
        yi2 = min(y12, y22)

        wi = max(0.0, xi2 - xi1)
        hi = max(0.0, yi2 - yi1)
        inter = wi * hi

        a1 = b1[2] * b1[3]
        a2 = b2[2] * b2[3]
        union = a1 + a2 - inter

        return inter / union if union > 0 else 0.0
