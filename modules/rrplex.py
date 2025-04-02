from enum import Enum
from .recorder import Recorder
from .replay import Replay


class RRPlexMode(Enum):
    BYPASS = 1
    RECORD = 2
    REPLAY = 3


def recordReplayMultiplex(childModule, mode):
    if mode == RRPlexMode.BYPASS:
        return childModule

    if mode == RRPlexMode.RECORD:
        return Recorder(childModule)

    if mode == RRPlexMode.REPLAY:
        return Replay(childModule.name)
