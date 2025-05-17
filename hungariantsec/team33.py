import cv2
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import KMeans

#TODO 4 MULTITHREADING VE 2 AYNI VIDEODA ISLEME EKLENECEK

class TeamAssigner:
    def __init__(self, team_colors_file):
        self.team_colors = self.load_team_colors(team_colors_file)

    def load_team_colors(self, filename):
        """Load team colors from a JSON file."""
        with open(filename, 'r') as f:
            data = json.load(f)
        print(f"Loaded team colors from {filename}: {data}")
        return {
            1: np.array(data["team_colors"]["team_1"]),
            2: np.array(data["team_colors"]["team_2"])
        }

    def classify_player(self, image, debug=False):
        """Classify a cropped image into a team using KMeans clustering."""
        # Calculate cropping region (middle 2/3 of the top-half)
        top_half_height = int(image.shape[0] / 2)
        skip_height = int(top_half_height / 3)  # Skip the top 1/3 of the top-half
        crop_height = int(top_half_height * 2 / 3)  # Take the middle 2/3 of the top-half
        cropped_region = image[skip_height:skip_height + crop_height, :]  # From skip to 2/3

        # Flatten the cropped region and apply KMeans clustering
        image_2d = cropped_region.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1).fit(image_2d)
        labels = kmeans.labels_
        cluster_centers = kmeans.cluster_centers_

        # Identify the player cluster using corner analysis
        labels_2d = labels.reshape(cropped_region.shape[0], cropped_region.shape[1])
        corner_clusters = [
            labels_2d[0, 0], labels_2d[0, -1],
            labels_2d[-1, 0], labels_2d[-1, -1]
        ]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color_bgr = cluster_centers[player_cluster]

        # Convert player color to RGB for consistency
        player_color_rgb = cv2.cvtColor(np.uint8([[player_color_bgr]]), cv2.COLOR_BGR2RGB)[0, 0]

        # Calculate distances to team colors
        distances = {
            team_id: np.linalg.norm(player_color_rgb - color)
            for team_id, color in self.team_colors.items()
        }
        assigned_team = min(distances, key=distances.get)

        # Calculate classification confidence
        total_distance = sum(distances.values())
        confidence = 1 - distances[assigned_team] / total_distance

        if debug:
            self.debug_clustering(image, labels_2d, player_color_rgb, assigned_team, skip_height, crop_height)

        return assigned_team, confidence

    def debug_clustering(self, image, labels, player_color, assigned_team, skip_height, crop_height):
        """Show debugging visualization with rectangle overlay."""
        plt.figure(figsize=(15, 5))

        # Original image with rectangle overlay
        overlay_image = image.copy()
        cv2.rectangle(overlay_image, (0, skip_height), (image.shape[1], skip_height + crop_height), (0, 255, 0), 2)

        plt.subplot(1, 4, 1)
        plt.imshow(cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB))
        plt.title("Original Image (Cropped Area Highlighted)")
        plt.axis("off")

        # Clustering visualization
        plt.subplot(1, 4, 2)
        plt.imshow(labels, cmap="viridis")
        plt.title("KMeans Clustering")
        plt.axis("off")

        # Extracted player color
        plt.subplot(1, 4, 3)
        plt.imshow([[player_color / 255]])
        plt.title("Extracted Player Color")
        plt.axis("off")

        # Classified team color
        plt.subplot(1, 4, 4)
        team_color = self.team_colors[assigned_team]
        plt.imshow([[team_color / 255]])
        plt.title(f"Team {assigned_team} Color")
        plt.axis("off")

        plt.tight_layout()
        plt.show()


def process_video_with_json(video_path, json_path, output_json, team_assigner, show_first_n=3):
    cap = cv2.VideoCapture(video_path)

    with open(json_path, 'r') as f:
        frame_data = json.load(f)

    output_data = []
    objects_shown = 0  # Counter for shown objects

    for frame_entry in tqdm(frame_data, desc=f"Processing {os.path.basename(json_path)}"):
        frame_index = frame_entry["frame_index"]
        objects = frame_entry["objects"]

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame {frame_index} from {video_path}")
            continue

        for obj_idx, obj in enumerate(objects):
            bbox = obj["bbox"]
            x1, y1, x2, y2 = map(int, bbox)
            cropped = frame[y1:y2, x1:x2]

            # Classify the cropped image
            debug = objects_shown < show_first_n  # Debug only for the first N objects
            team_index, confidence = team_assigner.classify_player(cropped, debug=debug)

            # Append classification results
            obj["team_index"] = team_index
            obj["team_classification_confidence"] = confidence

            if debug:
                objects_shown += 1
                print(f"Frame {frame_index}, Object {obj_idx + 1}:")
                print(f"  Team Index: {team_index}, Confidence: {confidence:.2f}")

        output_data.append({
            "frame_index": frame_index,
            "objects": objects
        })

    cap.release()

    # Save the updated JSON
    with open(output_json, 'w') as f:
        json.dump(output_data, f, indent=4)
    print(f"Processed and saved to {output_json}")


if __name__ == "__main__":
    video_json_pairs = [
        ("video_rightlong.mp4", "right_non_intersections.json"),
    ]

    team_colors_file = "team_colors.json"
    team_assigner = TeamAssigner(team_colors_file)

    for video_path, json_path in video_json_pairs:
        output_json = json_path.replace(".json", "2.json")
        process_video_with_json(video_path, json_path, output_json, team_assigner, show_first_n=10)
