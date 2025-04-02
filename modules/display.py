import cv2 as cv
import copy


class Display:
    def __init__(self, historyBufferSize):
        self.name = "Display"
        self.historyBufferSize = historyBufferSize

        self.showDetections = False
        self.showOpticalFlow = False
        self.showTracks = True

    def start(self, data):
        self.currentFrame = 0
        self.singleStepMode = True
        self.history = []

    def stop(self, data):
        pass

    def step(self, data):
        if data["stopped"]:
            self.singleStepMode = True
        else:
            # Append data to history
            self.history.append(copy.deepcopy(data))

        # If we exceeded the history buffer, strip away the first entry (FIFO)
        if self.historyBufferSize > 0 and len(self.history) > self.historyBufferSize:
            self.history = self.history[1:]

        self.moduleResults = {}
        self.visualizationLoop()

        return self.moduleResults

    def drawStatusBar(self, frame, data):
        def drawStatusBox(x1, boxLabel, textLabel, active):
            color = (0, 255, 0)
            if not active:
                color = (0, 0, 255)

            fontSize = cv.getTextSize(boxLabel, cv.FONT_HERSHEY_DUPLEX, 0.5, 1)[0]
            cv.rectangle(frame, (x1, y1), (x1 + fontSize[0] + 8, y2), color, -1)
            cv.putText(
                frame,
                boxLabel,
                (x1 + 4, y2 - 8),
                cv.FONT_HERSHEY_DUPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            x1 += fontSize[0] + 8

            fontSize = cv.getTextSize(textLabel, cv.FONT_HERSHEY_DUPLEX, 0.5, 1)[0]
            cv.putText(
                frame,
                textLabel,
                (x1 + 4, y2 - 8),
                cv.FONT_HERSHEY_DUPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            x1 += fontSize[0] + 8
            return x1

        x1 = 0
        x2 = frame.shape[1]
        y1 = frame.shape[0] - 24
        y2 = frame.shape[0]

        copy = frame.copy()
        cv.rectangle(copy, (x1, y1), (x2, y2), (255, 255, 255), -1)
        cv.rectangle(copy, (x1, 0), (x2, 24), (255, 255, 255), -1)
        frame = cv.addWeighted(frame, 0.5, copy, 0.5, gamma=0)

        cv.putText(
            frame,
            f"#{data['counter']}",
            (16, y2 - 8),
            cv.FONT_HERSHEY_DUPLEX,
            0.5,
            (200, 0, 0),
            1,
        )

        x1 = 70
        x1 = drawStatusBox(x1, "+", "Forward", False)
        x1 = drawStatusBox(x1, "-", "Backward", False)
        x1 = drawStatusBox(x1, "ESC", "Exit", False)
        x1 = drawStatusBox(x1, "Space", "Single Step", False)
        x1 = drawStatusBox(x1, "Enter", "Toggle", False)

        y1, y2 = 0, 24
        x1 = 70
        x1 = drawStatusBox(x1, "D", "Detections", self.showDetections)
        x1 = drawStatusBox(x1, "O", "Optical Flow", self.showOpticalFlow)
        x1 = drawStatusBox(x1, "T", "Tracks", self.showTracks)
        return frame

    def drawDetections(self, frame, data):
        if "detections" in data:
            detections = data["detections"]
        else:
            detections = []

        if "classes" in data:
            classes = data["classes"]
        else:
            classes = []

        for index, (_x, _y, _w, _h) in enumerate(detections):
            cls = classes[index]
            if cls == 0:  ## Ball
                _w *= 2.0  # Draw the ball bigger
                _h *= 2.0
                color = (255, 255, 255)
            if cls == 2:
                color = (255, 0, 0)  # Player
            if cls == 1:  # GoalKeeper
                color = (0, 128, 255)
            if cls == 3:  # Referee
                color = (64, 64, 64)

            x, y, w, h = int(_x - _w / 2.0), int(_y - _h / 2.0), int(_w), int(_h)

            cv.rectangle(frame, (x, y), (x + w, y + h), color, 1)

        return frame

    def drawTracks(self, frame, data):
        import cv2 as cv

        if "tracks" in data:
            tracks = data["tracks"]
            velocities = data["trackVelocities"]
            ages = data["trackAge"]
            classes = data["trackClasses"]
            teamClasses = data.get("teamClasses", [0] * len(tracks))
            teamAColor = data.get("teamAColor", (255, 0, 0))
            teamBColor = data.get("teamBColor", (0, 255, 0))
        else:
            tracks = []
            velocities = []
            ages = []
            classes = []
            teamClasses = []
            teamAColor = (255, 0, 0)
            teamBColor = (0, 255, 0)

        for index, (_x, _y, _w, _h) in enumerate(tracks):
            cls = classes[index] if index < len(classes) else 0

            if cls == 0:
                _w *= 2.0
                _h *= 2.0
                color = (255, 255, 255)
            elif cls == 1:
                color = (0, 128, 255)
            elif cls == 2:
                if index < len(teamClasses):
                    team_assignment = teamClasses[index]
                    if team_assignment == 1:
                        color = teamAColor
                    elif team_assignment == -1:
                        color = teamBColor
                    else:
                        color = (0, 165, 255)
                else:
                    color = (0, 165, 255)
            elif cls == 3:
                color = (64, 64, 64)
            else:
                color = (255, 255, 255)

            x = int(_x - _w / 2.0)
            y = int(_y - _h / 2.0)
            w = int(_w)
            h = int(_h)

            footX = int(_x)
            footY = int(_y + _h / 2.0 + 4.0)
            velX = int(_x + 5.0 * velocities[index][0])
            velY = int(4.0 + _y + _h / 2.0 + 5.0 * velocities[index][1])

            cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv.circle(frame, (footX, footY), 3, color, -1)
            cv.line(frame, (footX, footY), (velX, velY), color, 1)

            boxLabel = f"{ages[index]}"
            fontSize = cv.getTextSize(boxLabel, cv.FONT_HERSHEY_DUPLEX, 0.5, 1)[0]
            cv.rectangle(
                frame,
                (footX - fontSize[0] // 2, footY + 8),
                (footX + fontSize[0] // 2, footY + 24),
                (16, 32, 128),
                -1,
            )
            cv.putText(
                frame,
                boxLabel,
                (footX - fontSize[0] // 2, footY + 20),
                cv.FONT_HERSHEY_DUPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

        return frame

    def drawOpticalFlow(self, frame, data):
        cx = frame.shape[1] / 2
        cy = frame.shape[0] / 2
        dx, dy = data["opticalFlow"]
        frame = cv.circle(frame, (int(cx), int(cy)), 4, (0, 0, 255), -1)
        frame = cv.line(
            frame,
            (int(cx), int(cy)),
            (int(cx + 15 * dx), int(cy + 15 * dy)),
            (0, 0, 255),
            thickness=2,
        )
        return frame

    def drawTeamColors(self, frame, data):
        if "teamAColor" in data and "teamBColor" in data:
            w = 32
            h = 24
            colorA = data["teamAColor"]
            colorB = data["teamBColor"]

            frame = cv.rectangle(frame, (16, 32), (16 + w, 32 + h), colorA, -1)
            frame = cv.rectangle(
                frame, (16 + w + 8, 32), (16 + 2 * w + 8, 32 + h), colorB, -1
            )

        return frame

    def visualizeFrame(self, currentFrame):
        # Get the data to visualize)
        data = self.history[len(self.history) - currentFrame - 1]

        # Copy over the frame to draw
        frame = data["image"].copy()

        # Draw status bar at bottom
        frame = self.drawStatusBar(frame, data)
        if self.showDetections:
            frame = self.drawDetections(frame, data)
        if self.showOpticalFlow:
            frame = self.drawOpticalFlow(frame, data)
        if self.showTracks:
            frame = self.drawTracks(frame, data)

        frame = self.drawTeamColors(frame, data)

        # Show it
        cv.imshow("Main Window", frame)

    def processKey(self, key):
        # Abort loop on ESCAPE key (in fact, abort whole application)
        if key == 27:
            self.moduleResults["terminate"] = True
            return False

        # Step forward to next image on SPACE Key
        if key == 32:
            self.singleStepMode = True
            return False

        if key == 13:
            self.singleStepMode = not self.singleStepMode

        if key == 45:  # Minus
            if self.currentFrame < len(self.history) - 1:
                self.singleStepMode = True
                self.currentFrame += 1
                # print(self.currentFrame)

            return True

        if key == 43:  # Plus
            if self.currentFrame > 0:
                self.singleStepMode = True
                self.currentFrame = self.currentFrame - 1
                # print(self.currentFrame)

            return True

        if not self.singleStepMode:
            if self.currentFrame > 0:
                self.currentFrame -= 1
                return True

        if key == 100:
            self.showDetections = not self.showDetections
            if self.showDetections:
                self.showTracks = False

        if key == 116:
            self.showTracks = not self.showTracks
            if self.showTracks:
                self.showDetections = False

        if key == 111:
            self.showOpticalFlow = not self.showOpticalFlow

        return True

    def visualizationLoop(self):
        # Always start visualization with the last frame

        while True:
            # Visualize the current frame
            self.visualizeFrame(self.currentFrame)

            # Sleep
            if self.singleStepMode:
                c = cv.waitKey()
            else:
                c = cv.waitKey(5)

            # Process key pressed
            if not self.processKey(c):
                break

            if not self.singleStepMode and self.currentFrame == 0:
                break
