import cv2
import numpy as np
import json
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}

    def get_clustering_model(self, image):
        image_2d = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)
        return kmeans

    def get_player_color(self, image):
        # Crop a region (from 1/6th to 1/2 of the image height)
        top_half_image = image[int(image.shape[0] / 6):int(image.shape[0] / 2), :]

        # --- Debug displays removed so that we can show all final results together ---
        # (Optionally, you can uncomment these if you wish to see intermediate results.)
        # plt.figure()
        # plt.imshow(cv2.cvtColor(top_half_image, cv2.COLOR_BGR2RGB))
        # plt.title("Top-Half Image for Clustering")
        # plt.axis("off")
        # plt.show()

        # Perform clustering on the cropped image
        kmeans = self.get_clustering_model(top_half_image)
        labels = kmeans.labels_.reshape(top_half_image.shape[0], top_half_image.shape[1])

        # Debug display for clustering labels (removed for final composite display)
        # plt.figure()
        # plt.imshow(labels, cmap="viridis")
        # plt.title("Clustering Labels")
        # plt.axis("off")
        # plt.colorbar(label="Cluster")
        # plt.show()

        # Determine the player cluster by examining the image corners
        corner_clusters = [
            labels[0, 0],
            labels[0, -1],
            labels[-1, 0],
            labels[-1, -1],
        ]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        # Debug: Show the cluster centers (in BGR)
        print("Cluster Centers (RGB):", kmeans.cluster_centers_)

        # Return the extracted player color along with the clustering labels and original image.
        return kmeans.cluster_centers_[player_cluster], labels, image

    def assign_team_color(self, cropped_images):
        # Process each image and collect (player_color, clustering_labels, original_image)
        player_data = []  # List of tuples for each image.
        for idx, img in enumerate(cropped_images):
            pc, labels_img, orig = self.get_player_color(img)
            player_data.append((pc, labels_img, orig))
        
        # Extract only the player colors for clustering
        colors = [item[0] for item in player_data]

        # Cluster the extracted player colors into three teams
        kmeans = KMeans(n_clusters=3, init="k-means++", n_init=10)
        kmeans.fit(colors)
        cluster_labels = kmeans.labels_
        
        # Save team colors based on the clustering centers.
        # Here we map cluster label 0 to Team 1, label 1 to Team 2, and label 2 to Team 3.
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]
        self.team_colors[3] = kmeans.cluster_centers_[2]

        # Organize players by team. In case of multiple players per team, we pick the first one.
        teams = {0: None, 1: None, 2: None}
        for i, (pc, labels_img, orig) in enumerate(player_data):
            team = cluster_labels[i]
            if teams[team] is None:
                teams[team] = (pc, labels_img, orig)

        # Create a composite display: 3 rows (teams) x 3 columns (team color, clustering labels, original image)
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        for team_label in range(3):
            team_number = team_label + 1  # Mapping: 0 -> Team 1, etc.
            # Column 1: Display team color (convert BGR to RGB and normalize)
            team_color_bgr = self.team_colors[team_number]
            team_color_rgb = team_color_bgr[::-1] / 255.0
            axes[team_label, 0].imshow([[team_color_rgb]])
            axes[team_label, 0].set_title(f"Team {team_number} Color")
            axes[team_label, 0].axis("off")

            if teams[team_label] is not None:
                _, clustering_labels, orig_img = teams[team_label]
                # Column 2: Display clustering labels (use a colormap)
                axes[team_label, 1].imshow(clustering_labels, cmap="viridis")
                axes[team_label, 1].set_title("Clustering Labels")
                axes[team_label, 1].axis("off")
                # Column 3: Display the original image (convert from BGR to RGB)
                orig_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
                axes[team_label, 2].imshow(orig_rgb)
                axes[team_label, 2].set_title("Original Image")
                axes[team_label, 2].axis("off")
            else:
                axes[team_label, 1].axis("off")
                axes[team_label, 2].axis("off")

        plt.tight_layout()
        plt.show()

    def save_team_colors(self, filename):
        """Save team colors to a JSON file."""
        data = {
            "team_colors": {
                "team_1": self.team_colors[1].tolist(),
                "team_2": self.team_colors[2].tolist(),
                "team_3": self.team_colors[3].tolist()
            }
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Team colors saved to {filename}")


# ---------------- Main Code ----------------
# Initialize TeamAssigner
team_assigner = TeamAssigner()


# Load cropped player images (modify file names as needed)
image1 = cv2.imread('67.jpg')  # Team 1 image
image2 = cv2.imread('78.jpg')  # Team 2 image
image3 = cv2.imread('image.png')  # Team 3 image

# Process all three images
team_assigner.assign_team_color([image1, image2, image3])

# Save team colors to a file
team_assigner.save_team_colors("team_colors.json")

# Retrieve and print team colors
print(f"Team 1 Color (RGB): {team_assigner.team_colors[1]}")
print(f"Team 2 Color (RGB): {team_assigner.team_colors[2]}")
print(f"Team 3 Color (RGB): {team_assigner.team_colors[3]}")
