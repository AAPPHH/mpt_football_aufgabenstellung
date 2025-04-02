from re import sub
import os
import pickle

RECORDING_FOLDER = "record"


def camel_case(s):
    s = sub(r"(_|-)+", " ", s).title().replace(" ", "")
    return "".join([s[0].lower(), s[1:]])


class Replay:
    def __init__(self, moduleName):
        self.name = f"Replay ({moduleName})"
        self.moduleName = camel_case(moduleName)

    def start(self, data):
        videoName = data["video"].split("/")[-1].split(".")[0]
        self.filenamePrefix = f"{videoName}_{self.moduleName}"
        self.filenamePrefix = os.path.join(RECORDING_FOLDER, self.filenamePrefix)

        with open(f"{self.filenamePrefix}.pickle", "rb") as f:
            self.ledger = pickle.load(f)

        self.currentIndex = 0
        pass

    def step(self, data):
        if self.currentIndex < len(self.ledger):
            result = self.ledger[self.currentIndex]
        else:
            result = {}
        self.currentIndex += 1

        return result

    def stop(self, data):
        pass
