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
        # Crop top-half of the image
        top_half_image = image[0:int(image.shape[0] / 2), :]

        # Debugging: Show the top-half image
        plt.figure()
        plt.imshow(cv2.cvtColor(top_half_image, cv2.COLOR_BGR2RGB))
        plt.title("Top-Half Image for Clustering")
        plt.axis("off")
        plt.show()

        # Perform clustering
        kmeans = self.get_clustering_model(top_half_image)
        labels = kmeans.labels_.reshape(top_half_image.shape[0], top_half_image.shape[1])

        # Debugging: Show clustering labels as an image
        plt.figure()
        plt.imshow(labels, cmap="viridis")
        plt.title("Clustering Labels")
        plt.axis("off")
        plt.colorbar(label="Cluster")
        plt.show()

        # Find the player cluster (non-background)
        corner_clusters = [
            labels[0, 0],
            labels[0, -1],
            labels[-1, 0],
            labels[-1, -1],
        ]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        # Debugging: Show cluster centers
        print("Cluster Centers (RGB):", kmeans.cluster_centers_)

        return kmeans.cluster_centers_[player_cluster]

    def assign_team_color(self, cropped_images):
        player_colors = []
        for idx, img in enumerate(cropped_images):
            player_color = self.get_player_color(img)
            player_colors.append(player_color)

            # Debugging: Show the extracted player color
            plt.figure()
            plt.imshow([[player_color / 255]])
            plt.title(f"Player {idx + 1} Color")
            plt.axis("off")
            plt.show()

        # Cluster player colors into two teams
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors)

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

        # Debugging: Show team colors
        plt.figure()
        plt.imshow([[self.team_colors[1] / 255, self.team_colors[2] / 255]])
        plt.title("Assigned Team Colors")
        plt.axis("off")
        plt.show()

    def save_team_colors(self, filename):
        """Save team colors to a JSON file."""
        data = {
            "team_colors": {
                "team_1": self.team_colors[1].tolist(),
                "team_2": self.team_colors[2].tolist()
            }
        }
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Team colors saved to {filename}")


# Initialize TeamAssigner
team_assigner = TeamAssigner()

# Load cropped player images
image1 = cv2.imread('67.jpg')  # Replace with the path to the first cropped image
image2 = cv2.imread('78.jpg')  # Replace with the path to the second cropped image

# Assign team colors using the two cropped images
team_assigner.assign_team_color([image1, image2])

# Save team colors to a file
team_assigner.save_team_colors("team_colors.json")

# Retrieve and print team colors
print(f"Team 1 Color (RGB): {team_assigner.team_colors[1]}")
print(f"Team 2 Color (RGB): {team_assigner.team_colors[2]}")
