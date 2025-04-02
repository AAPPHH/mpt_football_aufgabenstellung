from re import sub
import os
import pickle

RECORDING_FOLDER = "record"


def camel_case(s):
    s = sub(r"(_|-)+", " ", s).title().replace(" ", "")
    return "".join([s[0].lower(), s[1:]])


class Recorder:
    def __init__(self, childModule):
        self.name = f"Recorder ({childModule.name})"
        self.child = childModule

        if not os.path.exists(RECORDING_FOLDER):
            os.mkdir(RECORDING_FOLDER)

    def start(self, data):
        videoName = data["video"].split("/")[-1].split(".")[0]
        moduleName = camel_case(self.child.name)
        self.filenamePrefix = f"{videoName}_{moduleName}"
        self.filenamePrefix = os.path.join(RECORDING_FOLDER, self.filenamePrefix)
        self.ledger = []

        self.child.start(data)
        pass

    def step(self, data):
        result = self.child.step(data)
        self.ledger.append(result)

        return result

    def stop(self, data):
        with open(f"{self.filenamePrefix}.pickle", "wb") as f:
            pickle.dump(self.ledger, f)
