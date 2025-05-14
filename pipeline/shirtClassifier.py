import numpy as np
import cv2
from collections import deque

class ShirtClassifier:
    """
    Shirt color classifier with adaptive calibration and robust labeling.

    This classifier uses histogram peak detection in Lab ab space, feature fusion
    of Lab ab and HSV, Mahalanobis distance classification, confidence thresholding,
    and two-level smoothing (median window + EMA). An adaptive inner-crop focuses
    on cleaner shirt pixels for improved stability under varying lighting.

    Attributes:
        calib_frames (int): Number of frames to collect before calibration.
        bins (int): Number of bins for the 2D Lab ab histogram.
        min_peak_dist (float): Minimum distance between calibration peaks.
        tau (float): Confidence threshold for uncertain labels.
        smoothing_window (int): Window size for median smoothing.
        ema_alpha (float): Alpha parameter for EMA smoothing.
        eps (float): Small value added for numerical stability.
        calibrated (bool): Whether calibration has been completed.
        peaks (list[np.ndarray] or None): Cluster center means after calibration.
        inv_covs (list[np.ndarray] or None): Inverse covariance matrices of clusters.
        track_windows (dict[int, deque]): Historical label windows per track ID.
        track_ema (dict[int, float]): EMA-smoothed label values per track ID.
    """
    def __init__(self,
                 calib_frames=150,
                 bins=32,
                 min_peak_dist=5.0,
                 tau=12.0,
                 smoothing_window=9,
                 ema_alpha=0.9,
                 eps=1e-5):
        """
        Initialize a ShirtClassifier instance.

        Args:
            calib_frames (int): Number of frames to collect features before running calibration.
            bins (int): Number of bins per dimension for the 2D Lab ab histogram.
            min_peak_dist (float): Minimum Euclidean distance between two calibration peaks in Lab ab.
            tau (float): Confidence threshold for Mahalanobis distance difference.
            smoothing_window (int): Size of the median filter window for label smoothing.
            ema_alpha (float): Smoothing factor (alpha) for exponential moving average.
            eps (float): Small epsilon added to covariance diagonal for numerical stability.
        """
        self.name = "ShirtClassifier"
        self.calib_frames = calib_frames
        self.bins = bins
        self.min_peak_dist = min_peak_dist
        self.tau = tau
        self.smoothing_window = smoothing_window
        self.ema_alpha = ema_alpha
        self.eps = eps
        self._cal_data = []
        self.calibrated = False
        self.peaks = None
        self.inv_covs = None
        self.track_windows = {}
        self.track_ema = {}

    def start(self, data):
        """
        Notify that calibration has not yet started.

        Args:
            data (dict): Optional data for start event (ignored).
        """
        print("[INFO] ShirtClassifier: Calibration not started.")

    def stop(self, data):
        """
        Notify that the classifier has stopped.

        Args:
            data (dict): Optional data for stop event (ignored).
        """
        print("[INFO] ShirtClassifier: Stopped.")

    def _calibrate_peaks(self):
        """
        Perform calibration by finding two dominant color peaks in Lab ab space
        and computing their means and inverse covariances.

        This method updates `self.peaks`, `self.inv_covs`, and sets `self.calibrated`.
        """
        if not self._cal_data:
            return
        arr = np.vstack(self._cal_data)  # shape (N, 5)
        ab = arr[:, :2]
        H, a_edges, b_edges = np.histogram2d(ab[:, 0], ab[:, 1], bins=self.bins)
        flat = H.flatten()
        idxs = np.argsort(flat)[::-1]
        peaks_ab = []
        for idx in idxs:
            if flat[idx] <= 0:
                break
            ai, bi = divmod(idx, self.bins)
            a_mid = (a_edges[ai] + a_edges[ai+1]) / 2.0
            b_mid = (b_edges[bi] + b_edges[bi+1]) / 2.0
            peaks_ab.append(np.array([a_mid, b_mid]))
            if len(peaks_ab) == 2:
                break
        if len(peaks_ab) < 2 or np.linalg.norm(peaks_ab[0] - peaks_ab[1]) < self.min_peak_dist:
            print("[WARN] Calibration peaks invalid, continue collecting.")
            return

        dists = np.linalg.norm(ab[:, None] - np.array(peaks_ab)[None, :], axis=2)
        labels = np.argmin(dists, axis=1)
        peaks_full, inv_covs = [], []
        for i in [0, 1]:
            pts = arr[labels == i]
            if pts.shape[0] < 2:
                print(f"[WARN] Cluster {i} too small; skipping calibration.")
                return
            mu = pts.mean(axis=0)
            cov = np.cov(pts, rowvar=False) + self.eps * np.eye(pts.shape[1])
            inv_covs.append(np.linalg.inv(cov))
            peaks_full.append(mu)
        self.peaks = peaks_full
        self.inv_covs = inv_covs
        self.calibrated = True
        print("[INFO] Calibration complete.")
        self._cal_data.clear()

    def step(self, data):
        """
        Classify each tracked object into Team A or Team B based on shirt color.

        Computes features in Lab and HSV, applies Mahalanobis classification,
        and smooths labels over time.

        Args:
            data (dict):
                image (np.ndarray): BGR frame.
                tracks (np.ndarray): Array of bounding boxes [x_center, y_center, w, h].
                trackIds (list[int]): Unique identifiers for each track.
                trackClasses (list[int]): Class IDs (players=1,2).

        Returns:
            dict: {
                'teamAColor': (B, G, R) tuple for Team A,
                'teamBColor': (B, G, R) tuple for Team B,
                'teamClasses': list of labels per track (1=Team A, -1=Team B, 0=undecided)
            }
        """
        frame = data.get("image")
        tracks = data.get("tracks", np.zeros((0, 4)))
        track_ids = data.get("trackIds", list(range(len(tracks))))
        track_classes = data.get("trackClasses", [])
        team_classes = [0] * len(tracks)
        teamA_color = (0, 0, 255)
        teamB_color = (255, 0, 0)

        if frame is None or len(tracks) < 2:
            return {"teamAColor": teamA_color,
                    "teamBColor": teamB_color,
                    "teamClasses": team_classes}

        feats, idxs = [], []
        h, w = frame.shape[:2]

        for i, (xc, yc, tw, th) in enumerate(tracks):
            cls = track_classes[i] if i < len(track_classes) else 0
            if cls not in [1, 2]:
                continue
            x1 = max(0, int(xc - tw/2)); y1 = max(0, int(yc - th/2))
            x2 = min(w-1, int(xc + tw/2)); y2 = min(h-1, int(yc + th/2))
            if x2 <= x1 or y2 <= y1:
                continue
            bw = x2 - x1
            border_x = min(max(2, int(0.07 * bw)), bw // 4)
            x1c, x2c = x1 + border_x, x2 - border_x
            bh = y2 - y1
            y1c = y1 + int(0.3 * bh)
            y2c = y1 + int(0.6 * bh)
            roi = frame[y1c:y2c, x1c:x2c]
            if roi.size == 0:
                continue
            lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            ab = lab[..., 1:3].reshape(-1, 2)
            h_vals = hsv[..., 0].astype(np.float32) / 180.0 * 2 * np.pi
            cos_h = np.cos(h_vals).reshape(-1)
            sin_h = np.sin(h_vals).reshape(-1)
            s = hsv[..., 1].astype(np.float32) / 255.0
            feat = np.array([
                np.median(ab[:, 0]), np.median(ab[:, 1]),
                np.median(cos_h), np.median(sin_h), np.median(s)
            ], dtype=np.float32)
            feats.append(feat)
            idxs.append(i)
            if not self.calibrated:
                self._cal_data.append(feat)

        if not self.calibrated and len(self._cal_data) >= self.calib_frames:
            self._calibrate_peaks()

        if not self.calibrated:
            return {"teamAColor": teamA_color,
                    "teamBColor": teamB_color,
                    "teamClasses": team_classes}

        for j, feat in enumerate(feats):
            i = idxs[j]
            tid = track_ids[i]
            d0 = (feat - self.peaks[0]) @ self.inv_covs[0] @ (feat - self.peaks[0])
            d1 = (feat - self.peaks[1]) @ self.inv_covs[1] @ (feat - self.peaks[1])
            conf = abs(d0 - d1)
            raw_label = 0 if conf < self.tau else (1 if d0 < d1 else -1)

            if tid not in self.track_windows:
                self.track_windows[tid] = deque(maxlen=self.smoothing_window)
                self.track_ema[tid] = raw_label
            self.track_windows[tid].append(raw_label)

            nonzero = [x for x in self.track_windows[tid] if x != 0]
            if len(nonzero) < 3:
                assign = 0
            else:
                med = np.median(self.track_windows[tid])
                old = self.track_ema.get(tid, med)
                ema = self.ema_alpha * old + (1 - self.ema_alpha) * med
                self.track_ema[tid] = ema
                assign = 1 if ema >= 0 else -1

            team_classes[i] = assign

        active = set(track_ids)
        for tid in list(self.track_windows):
            if tid not in active:
                self.track_windows.pop(tid)
                self.track_ema.pop(tid, None)

        return {"teamAColor": teamA_color,
                "teamBColor": teamB_color,
                "teamClasses": team_classes}
