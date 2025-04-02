import numpy as np
from sklearn.cluster import KMeans

class ShirtClassifier:
    """
    A classifier that splits player shirt colors into two clusters using K-Means (n_clusters=2).
    Maintains consistency such that:
    - Team A is always assigned red (BGR = (0, 0, 255))
    - Team B is always assigned blue (BGR = (255, 0, 0))
    """

    def __init__(self):
        """
        Initialize the ShirtClassifier.

        Attributes:
            name (str): Name of the module.
            reference_colors (ndarray or None): Cluster centers from the first 'good' frame (if any).
            reference_labels (dict or None): A dictionary mapping cluster labels to specific teams.
        """
        self.name = "Shirt Classifier"
        self.reference_colors = None  # Save cluster centers from first good frame
        self.reference_labels = None  # Save label to team mapping

    def start(self, data):
        """
        Called when the classifier is started.

        Args:
            data (dict): A dictionary containing relevant data for the start event
                         (can be empty or contain system-specific parameters).
        """
        print("[INFO] Shirt Classifier wurde gestartet.")

    def stop(self, data):
        """
        Called when the classifier is stopped.

        Args:
            data (dict): A dictionary containing relevant data for the stop event
                         (can be empty or contain system-specific parameters).
        """
        print("[INFO] Shirt Classifier wurde gestoppt.")

    def step(self, data):
        """
        Perform a single step of the classification process by splitting players into two teams
        (red or blue) based on average shirt color.

        The method does the following:
        1. Retrieves player bounding boxes (tracks) from the data.
        2. Filters out objects that are not relevant (class not 1 or 2).
        3. Computes the average color of the top half of each player's bounding box.
        4. Applies K-Means (k=2) to cluster these average colors.
        5. Assigns one cluster to "Team A" (red) and the other cluster to "Team B" (blue).

        Args:
            data (dict): A dictionary containing:
                - 'image' (numpy.ndarray): The current frame in BGR format.
                - 'tracks' (ndarray): An array of bounding boxes in (x_center, y_center, w, h) format.
                - 'trackClasses' (list): A list of class IDs corresponding to each track.

        Returns:
            dict: A dictionary with the following keys:
                - 'teamAColor': Tuple of (B, G, R) for team A.
                - 'teamBColor': Tuple of (B, G, R) for team B.
                - 'teamClasses': A list with the team assignment for each tracked object:
                    0 = Not decided / not a player
                    1 = Player belongs to team A (red)
                    -1 = Player belongs to team B (blue)
        """
        frame = data.get("image", None)
        tracks = data.get("tracks", np.zeros((0, 4)))
        track_classes = data.get("trackClasses", [])

        # Fixed team colors (BGR)
        team_a_color = (0, 0, 255)  # Red
        team_b_color = (255, 0, 0)  # Blue

        team_classes = [0] * len(tracks)

        if frame is None or len(tracks) == 0:
            return {
                "teamAColor": team_a_color,
                "teamBColor": team_b_color,
                "teamClasses": team_classes,
            }

        height, width, _ = frame.shape
        player_colors = []
        player_indices = []

        for i, (x_center, y_center, w, h) in enumerate(tracks):
            cls = track_classes[i] if i < len(track_classes) else 0
            if cls not in [1, 2]:  # If it's not a player
                team_classes[i] = 0
                continue

            x1 = int(x_center - w / 2)
            y1 = int(y_center - h / 2)
            x2 = int(x_center + w / 2)
            y2 = int(y_center + h / 2)

            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width - 1, x2)
            y2 = min(height - 1, y2)

            if x2 <= x1 or y2 <= y1:
                team_classes[i] = 0
                continue

            # Take top half of the bounding box to focus on the shirt region
            tshirt_y2 = y1 + (y2 - y1) // 2
            roi = frame[y1:tshirt_y2, x1:x2]
            if roi.size == 0:
                team_classes[i] = 0
                continue

            avg_color = roi.mean(axis=(0, 1))  # (B, G, R) format
            player_colors.append(avg_color)
            player_indices.append(i)

        # If there are fewer than 2 players to cluster, we can't reliably split them into teams
        if len(player_colors) < 2:
            return {
                "teamAColor": team_a_color,
                "teamBColor": team_b_color,
                "teamClasses": team_classes,
            }

        X = np.array(player_colors, dtype=np.float32)
        kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X)
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_

        # Desired colors for teams (BGR)
        desired_team_a_color = np.array([0, 0, 255], dtype=np.float32)  # Red
        desired_team_b_color = np.array([255, 0, 0], dtype=np.float32)  # Blue

        dist_to_a = np.linalg.norm(centers - desired_team_a_color, axis=1)
        dist_to_b = np.linalg.norm(centers - desired_team_b_color, axis=1)

        # If each center is closer to a unique desired color, use that mapping.
        # Otherwise, default to cluster 0 = Team A, cluster 1 = Team B.
        if np.argmin(dist_to_a) != np.argmin(dist_to_b):
            cluster_for_a = int(np.argmin(dist_to_a))
            cluster_for_b = int(np.argmin(dist_to_b))
            label_map = {
                cluster_for_a: 1,   # Team A (red)
                cluster_for_b: -1,  # Team B (blue)
            }
        else:
            label_map = {0: 1, 1: -1}

        for idx, cluster_label in enumerate(labels):
            track_index = player_indices[idx]
            team_classes[track_index] = label_map[cluster_label]

        result = {
            "teamAColor": team_a_color,
            "teamBColor": team_b_color,
            "teamClasses": team_classes,
        }

        return result
