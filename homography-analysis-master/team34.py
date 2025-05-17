import cv2
import numpy as np
import json
import os
import glob
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

# ---------------- Helper Function ----------------
def extract_player_info(image, debug=False):
    """
    Extract the dominant player color, clustering labels, and return the original image.
    The region is cropped from 1/6 to 1/2 of the image height.
    """
    # Crop the region for analysis
    h = image.shape[0]
    crop = image[int(h / 6):int(h / 2), :]
    
    # Flatten and perform KMeans clustering (2 clusters)
    pixels = crop.reshape(-1, 3)
    kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1, random_state=42).fit(pixels)
    labels = kmeans.labels_.reshape(crop.shape[0], crop.shape[1])
    
    # Determine the player cluster using the image corners
    corners = [labels[0, 0], labels[0, -1], labels[-1, 0], labels[-1, -1]]
    non_player = max(set(corners), key=corners.count)
    player_cluster = 1 - non_player
    dominant_color = kmeans.cluster_centers_[player_cluster]
    
    # Optional debugging display (disabled by default)
    if debug:
        plt.figure()
        plt.imshow(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        plt.title("Cropped Region")
        plt.axis("off")
        plt.show()
        
        plt.figure()
        plt.imshow(labels, cmap="viridis")
        plt.title("Clustering Labels")
        plt.axis("off")
        plt.colorbar()
        plt.show()
    
    return dominant_color, labels, image

# ---------------- Load Team Colors from JSON ----------------
with open("team_colors.json", "r") as f:
    team_colors_json = json.load(f)["team_colors"]

# Convert team colors to NumPy arrays (they are in BGR order)
team1_color = np.array(team_colors_json["team_1"])
team2_color = np.array(team_colors_json["team_2"])
team3_color = np.array(team_colors_json["team_3"])

# ---------------- Load and Process Images ----------------
# Get all image file paths from the images directory (assuming common image extensions)
image_files = glob.glob(os.path.join("images", "*.[jp][pn]g"))  # matches .jpg, .png, etc.
# Make sure to select exactly 21 images (sorted alphabetically)
image_files = sorted(image_files)[:21]

# Process each image: extract dominant color, clustering labels, and original image.
images_info = []  # List of dicts for each image
for file in image_files:
    img = cv2.imread(file)
    if img is None:
        print(f"Warning: Unable to load {file}")
        continue
    dominant_color, labels, orig = extract_player_info(img)
    images_info.append({
        "file": file,
        "dominant_color": dominant_color,
        "labels": labels,
        "orig": orig
    })

if len(images_info) != 21:
    raise ValueError("Expected 21 valid images in the images directory.")

# ---------------- Classify Team 3 ----------------
# Compute distance from each image's dominant color to team 3 color center
dists_to_team3 = [np.linalg.norm(info["dominant_color"] - team3_color) for info in images_info]
idx_team3 = np.argmin(dists_to_team3)
team3_info = images_info.pop(idx_team3)  # Select and remove the image that best matches team 3

# ---------------- Linear Sum Assignment for Teams 1 & 2 ----------------
# Now we have 20 images remaining.
num_remaining = len(images_info)  # should be 20

# Create a cost matrix: 20 images x 20 slots (first 10 slots for team 1, next 10 for team 2)
cost = np.zeros((num_remaining, 20))
for i, info in enumerate(images_info):
    d1 = np.linalg.norm(info["dominant_color"] - team1_color)
    d2 = np.linalg.norm(info["dominant_color"] - team2_color)
    cost[i, :10] = d1  # For team 1 slots
    cost[i, 10:] = d2  # For team 2 slots

# Solve the assignment problem (Hungarian algorithm)
row_ind, col_ind = linear_sum_assignment(cost)

# Assign images to teams 1 and 2 based on the slot index.
team1_images = []
team2_images = []
for r, c in zip(row_ind, col_ind):
    if c < 10:
        team1_images.append(images_info[r])
    else:
        team2_images.append(images_info[r])

print(f"Assigned {len(team1_images)} images to Team 1 and {len(team2_images)} images to Team 2.")
print("Team 3 has 1 image.")

# ---------------- Display Results ----------------
def display_team_results(team_number, team_color, team_images, images_per_row=5):
    """
    For each image in team_images, display a row with:
      Column 1: The team color (color block)
      Column 2: The clustering labels image (from the cropped region)
      Column 3: The original image (converted to RGB)
    """
    n = len(team_images)
    rows = (n + images_per_row - 1) // images_per_row  # Calculate required rows
    
    fig, axes = plt.subplots(rows, 3 * images_per_row, figsize=(12, 4 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)  # Ensure axes is 2D for consistency

    # Iterate through each image
    for i, info in enumerate(team_images):
        row_idx = i // images_per_row  # Calculate row index
        col_idx = (i % images_per_row) * 3  # Calculate column index (3 columns per image)
        
        # Column 1: Team color block (convert BGR to RGB and normalize to [0,1])
        team_color_rgb = team_color[::-1] / 255.0
        axes[row_idx, col_idx].imshow([[team_color_rgb]])
        axes[row_idx, col_idx].set_title(f"Team {team_number} Color")
        axes[row_idx, col_idx].axis("off")
        
        # Column 2: Clustering labels image
        axes[row_idx, col_idx + 1].imshow(info["labels"], cmap="viridis")
        axes[row_idx, col_idx + 1].set_title("Clustering Labels")
        axes[row_idx, col_idx + 1].axis("off")
        
        # Column 3: Original image (convert from BGR to RGB)
        orig_rgb = cv2.cvtColor(info["orig"], cv2.COLOR_BGR2RGB)
        axes[row_idx, col_idx + 2].imshow(orig_rgb)
        axes[row_idx, col_idx + 2].set_title("Original Image")
        axes[row_idx, col_idx + 2].axis("off")
    
    # Remove unused axes if any
    for j in range(i + 1, rows * images_per_row):
        row_idx = j // images_per_row
        col_idx = (j % images_per_row) * 3
        axes[row_idx, col_idx].axis("off")
        axes[row_idx, col_idx + 1].axis("off")
        axes[row_idx, col_idx + 2].axis("off")

    plt.tight_layout()
    plt.show()


# Display results for Team 1, Team 2, and Team 3.
if team1_images:
    display_team_results(1, team1_color, team1_images)
if team2_images:
    display_team_results(2, team2_color, team2_images)
# Team 3 has one image:
display_team_results(3, team3_color, [team3_info])
