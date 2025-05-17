import cv2
import numpy as np
import json

# Video paths
VIDEO_LEFT = "l5.mp4"
VIDEO_RIGHT = "r5.mp4"
OUTPUT_VIDEO = "output.mp4"
POINTS_FILE = "points.json"
JSON_INPUT = "input2.json"
JSON_OUTPUT = "output.json"

# Resize dimensions
WIDTH, HEIGHT = 750, 500
# How many pixels to shift the right video upward when merging:
SHIFT_UP = 10  # ← change this to whatever you need

N_FRAMES = 5000  # Process only the first N frames

# Load videos
cap_left = cv2.VideoCapture(VIDEO_LEFT)
cap_right = cv2.VideoCapture(VIDEO_RIGHT)

# Get video properties
fps = int(cap_left.get(cv2.CAP_PROP_FPS))
orig_width_left = int(cap_left.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_height_left = int(cap_left.get(cv2.CAP_PROP_FRAME_HEIGHT))
orig_width_right = int(cap_right.get(cv2.CAP_PROP_FRAME_WIDTH))
orig_height_right = int(cap_right.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Calculate scaling factors
scale_x_left = WIDTH / orig_width_left
scale_y_left = HEIGHT / orig_height_left
scale_x_right = WIDTH / orig_width_right
scale_y_right = HEIGHT / orig_height_right

# Video writer setup
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (WIDTH * 2, HEIGHT))

# Process video frames
frames = []
for _ in range(N_FRAMES):
    ret1, frame1 = cap_left.read()
    ret2, frame2 = cap_right.read()
    
    if not ret1 or not ret2:
        break
    
    # Resize frames
    frame1 = cv2.resize(frame1, (WIDTH, HEIGHT))
    frame2 = cv2.resize(frame2, (WIDTH, HEIGHT))
    
    # Shift the right frame up by SHIFT_UP pixels
    frame2_shifted = np.zeros_like(frame2)
    if 0 <= SHIFT_UP < HEIGHT:
        frame2_shifted[0:HEIGHT-SHIFT_UP, :] = frame2[SHIFT_UP:HEIGHT, :]
    else:
        # if SHIFT_UP out of range, just use the unshifted frame
        frame2_shifted[:] = frame2
    
    # Concatenate frames side by side
    combined_frame = np.hstack((frame1, frame2_shifted))
    frames.append(combined_frame)

cap_left.release()
cap_right.release()

# Extract middle frame for point selection
middle_frame = frames[len(frames) // 2].copy()

# List to store selected points
selected_points = []

def select_points(event, x, y, flags, param):
    """Mouse callback to select points."""
    global selected_points
    if event == cv2.EVENT_LBUTTONDOWN:
        selected_points.append((x, y))
        print(f"Selected: {x}, {y}")

        # Draw the selected point
        cv2.circle(middle_frame, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Select 4 Points", middle_frame)

        # Stop when 4 points are selected
        if len(selected_points) == 4:
            cv2.destroyAllWindows()

# Show middle frame and get points
cv2.imshow("Select 4 Points", middle_frame)
cv2.setMouseCallback("Select 4 Points", select_points)
cv2.waitKey(0)

# Automatically determine point order
selected_points = sorted(selected_points, key=lambda p: (p[1], p[0]))  # Sort by Y first, then X
top_two = sorted(selected_points[:2], key=lambda p: p[0])  # Sort first two by X
bottom_two = sorted(selected_points[2:], key=lambda p: p[0])  # Sort last two by X

ordered_points = {
    "top_left": top_two[0],
    "top_right": top_two[1],
    "bottom_right": bottom_two[1],
    "bottom_left": bottom_two[0]
}

# Draw the quadrilateral on selected frames
polygon_points = np.array([list(ordered_points.values())], np.int32)
for i in range(len(frames)):
    overlay = frames[i].copy()
    cv2.polylines(overlay, [polygon_points], isClosed=True, color=(128, 0, 128), thickness=2)
    cv2.fillPoly(overlay, [polygon_points], color=(128, 0, 128))
    frames[i] = cv2.addWeighted(overlay, 0.3, frames[i], 0.7, 0)

# Write output video
for frame in frames:
    out.write(frame)
out.release()

# Save selected points to JSON
with open(POINTS_FILE, "w") as f:
    json.dump(ordered_points, f, indent=4)

# Load JSON file for bounding box scaling
with open(JSON_INPUT, "r") as f:
    data = json.load(f)

# Update bounding box coordinates
for frame in data["frames"][:N_FRAMES]:
    for obj in frame["obj"]:
        x1, y1, x2, y2 = obj["bbox"]
        
        # Determine which video the object belongs to using color
        if obj["color"] in ["blue", "purple"]:  # Left video
            obj["bbox"] = [
                x1 * scale_x_left,
                y1 * scale_y_left,
                x2 * scale_x_left,
                y2 * scale_y_left
            ]
        else:  # Right video
            # apply scaling...
            bx1 = x1 * scale_x_right
            by1 = y1 * scale_y_right
            bx2 = x2 * scale_x_right
            by2 = y2 * scale_y_right
            # …then subtract the vertical shift so the JSON y‑coords match the shifted video
            obj["bbox"] = [
                bx1,
                max(0, by1 - SHIFT_UP),
                bx2,
                max(0, by2 - SHIFT_UP)
            ]

# Save updated JSON
with open(JSON_OUTPUT, "w") as f:
    json.dump(data, f, indent=4)

print(f"Points saved to {POINTS_FILE}")
print(f"Updated JSON saved as {JSON_OUTPUT}")
print(f"Concatenated video saved as {OUTPUT_VIDEO}")
