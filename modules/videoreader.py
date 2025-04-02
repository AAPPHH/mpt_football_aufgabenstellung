import cv2 as cv


class VideoReader:
    def __init__(self, targetSize):
        self.targetSize = targetSize
        self.name = "VideoReader"

    def start(self, data):
        self.frameCounter = 0
        self.cap = cv.VideoCapture(data["video"])

    def stop(self, data):
        pass

    def step(self, signals):
        ret, frame = self.cap.read()
        if ret == False:
            return {"stopped": True}

        self.frameCounter = self.frameCounter + 1

        if self.targetSize is not None:
            frame = cv.resize(frame, self.targetSize)

        return {"counter": self.frameCounter, "image": frame, "stopped": False}
