import cv2 as cv
import numpy as np

class OpticalFlow:
    """
    A class for calculating the optical flow between consecutive frames using Farneback's method.
    It can optionally scale and mirror the input frames and supports CUDA-accelerated flow.
    """

    def __init__(self, scale_factor=0.5, mirror=True):
        """
        Initialize the optical flow computation with given parameters.

        Args:
            scale_factor (float): Scaling factor for faster processing
                                  (e.g., 0.5 for 50% size).
            mirror (bool): If True, horizontally flip the frame.
            use_gpu (bool): If True, use CUDA-based optical flow (requires an NVIDIA GPU).
        """
        self.name = "Optical Flow"
        self.prev_gray = None
        self.scale_factor = scale_factor
        self.mirror = mirror

    def start(self, data):
        """
        Called when the module is started. Resets any previous state.

        Args:
            data (dict): A dictionary containing relevant data for the start event 
                         (can be empty or system-specific).
        """
        print("[INFO] Optical Flow wurde gestartet.")
        self.prev_gray = None

    def stop(self, data):
        """
        Called when the module is stopped. Resets any previous state.

        Args:
            data (dict): A dictionary containing relevant data for the stop event
                         (can be empty or system-specific).
        """
        print("[INFO] Optical Flow wurde gestoppt.")
        self.prev_gray = None

    def step(self, data):
        """
        Process the current frame to calculate the optical flow.

        This method:
        1. Optionally resizes the frame by a given scale factor.
        2. Optionally flips the frame horizontally.
        3. Converts the frame to grayscale.
        4. Calculates the Farneback optical flow between this frame and the previous one.
        5. Returns the average flow vector in (x, y).

        Args:
            data (dict): A dictionary containing:
                - 'image' (numpy.ndarray): The current frame in BGR format.

        Returns:
            dict: A dictionary containing:
                - 'opticalFlow' (numpy.ndarray or None): The average optical flow vector.
                                                        None if the frame is missing.
        """
        frame = data.get("image", None)
        if frame is None:
            return {"opticalFlow": None}

        if self.scale_factor != 1.0:
            frame = cv.resize(
                frame,
                (0, 0),
                fx=self.scale_factor,
                fy=self.scale_factor,
                interpolation=cv.INTER_LINEAR,
            )

        if self.mirror:
            frame = cv.flip(frame, 1)

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return {"opticalFlow": np.array([0, 0])}

        flow = cv.calcOpticalFlowFarneback(
            self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        avg_flow = np.mean(flow, axis=(0, 1)) / self.scale_factor

        self.prev_gray = gray
        return {"opticalFlow": avg_flow}

