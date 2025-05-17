import json
import numpy as np
import cv2

# File paths
TRACKING_DATA_RIGHT = "right5.json"
TRACKING_DATA_LEFT = "left5shifted.json"
HOMOGRAPHY_MATRIX_LEFT = "al2_homography_matrix.txt"
HOMOGRAPHY_MATRIX_RIGHT = "al1_homography_matrix.txt"
DIMENSIONS_FILE = "dimensions.txt"
OUTPUT_RIGHT_JSON = "right_intersections.json"
OUTPUT_RIGHT_NON_INTERSECTIONS_JSON = "right_non_intersections.json"
OUTPUT_LEFT_JSON = "left_intersections.json"
OUTPUT_LEFT_NON_INTERSECTIONS_JSON = "left_non_intersections.json"

# Color mapping
COLORS = {
    "right_intersection": "orange",
    "right_non_intersection": "red",
    "left_intersection": "purple",
    "left_non_intersection": "blue",
}

def load_dimensions_and_homographies():
    """Load blue line positions and homography matrices."""
    with open(DIMENSIONS_FILE, "r") as f:
        lines = f.readlines()
    blue_line_right, width_right = map(int, lines[0].split())
    blue_line_left, width_left = map(int, lines[1].split())

    homography_matrix_left = np.loadtxt(HOMOGRAPHY_MATRIX_LEFT, delimiter=' ')
    homography_matrix_right = np.loadtxt(HOMOGRAPHY_MATRIX_RIGHT, delimiter=' ')

    return blue_line_right, width_right, blue_line_left, width_left, homography_matrix_left, homography_matrix_right

def transform_point(point, homography_matrix):
    """Transform a single point using a homography matrix."""
    point = np.array([[point]], dtype=np.float32)
    transformed_point = cv2.perspectiveTransform(point, homography_matrix)
    return transformed_point[0][0]

def is_point_within_bounds(point, frame_width, frame_height):
    """Check if a point is within the bounds of the transformed image."""
    x, y = point
    return 0 <= x < frame_width and 0 <= y < frame_height

from tqdm import tqdm  # Import tqdm

def filter_objects(data, homography, red_line, frame_width, frame_height, is_right):
    """
    Filter bounding boxes into intersection and non-intersection groups with color added.
    """
    intersections = []
    non_intersections = []

    for frame in tqdm(data, desc="Processing frames", total=len(data)):
        frame_index = frame["frame_index"]
        intersecting_objects = []
        non_intersecting_objects = []

        # Check bounding boxes
        for obj in frame.get("objects", []):
            bbox = obj["bbox"]
            bottom_middle = [(bbox[0] + bbox[2]) / 2, bbox[3]]  # Centroid width coordinate
            transformed_point = transform_point(bottom_middle, homography)
            x_trans, y_trans = transformed_point
            

            if is_right:
                if x_trans < 25:
                    continue  # Skip this object if it's beyond the red line on the right
                in_intersection = (25 < x_trans < red_line and is_point_within_bounds(transformed_point, frame_width, frame_height))
            else:
                if x_trans > frame_width - 25:
                    continue  # Skip this object if it's before the red line on the left
                in_intersection = (red_line < x_trans < frame_width - 25 and is_point_within_bounds(transformed_point, frame_width, frame_height))

            if in_intersection:
                obj["color"] = COLORS["right_intersection"] if is_right else COLORS["left_intersection"]
                intersecting_objects.append(obj)
            else:
                obj["color"] = COLORS["right_non_intersection"] if is_right else COLORS["left_non_intersection"]
                non_intersecting_objects.append(obj)

        if intersecting_objects:
            intersections.append({"frame_index": frame_index, "objects": intersecting_objects})
        if non_intersecting_objects:
            non_intersections.append({"frame_index": frame_index, "objects": non_intersecting_objects})

    return intersections, non_intersections

def main():
    # Load dimensions, homographies, and offset
    blue_line_right, width_right, blue_line_left, width_left, homography_left, homography_right = load_dimensions_and_homographies()

    # Offset for red line calculation
    offset_x = blue_line_right / 6 # First number from the first line of dimensions.txt

    # Compute red line indexes
    red_line_right = blue_line_right + offset_x
    red_line_left = blue_line_left - offset_x

    # Transformed image dimensions
    frame_width, frame_height = 400, 300

    # Load tracking data
    with open(TRACKING_DATA_RIGHT, "r") as f:
        data_right = json.load(f)
    with open(TRACKING_DATA_LEFT, "r") as f:
        data_left = json.load(f)

    # Filter objects for right and left videos
    right_intersections, right_non_intersections = filter_objects(
        data_right, homography_right, red_line_right, frame_width, frame_height, is_right=True
    )
    left_intersections, left_non_intersections = filter_objects(
        data_left, homography_left, red_line_left, frame_width, frame_height, is_right=False
    )

    # Save the filtered intersections and non-intersections
    with open(OUTPUT_RIGHT_JSON, "w") as right_file:
        json.dump(right_intersections, right_file, indent=2)
    with open(OUTPUT_RIGHT_NON_INTERSECTIONS_JSON, "w") as right_non_file:
        json.dump(right_non_intersections, right_non_file, indent=2)
    with open(OUTPUT_LEFT_JSON, "w") as left_file:
        json.dump(left_intersections, left_file, indent=2)
    with open(OUTPUT_LEFT_NON_INTERSECTIONS_JSON, "w") as left_non_file:
        json.dump(left_non_intersections, left_non_file, indent=2)

    print(f"Right intersections saved to {OUTPUT_RIGHT_JSON}")
    print(f"Right non-intersections saved to {OUTPUT_RIGHT_NON_INTERSECTIONS_JSON}")
    print(f"Left intersections saved to {OUTPUT_LEFT_JSON}")
    print(f"Left non-intersections saved to {OUTPUT_LEFT_NON_INTERSECTIONS_JSON}")

import filterjson2

if __name__ == "__main__":
    main()
    print("First script completed. Now running the second script...")
    filterjson2.main()  # Call the second script after finishing the first

def run_homography_and_merge(
    tracking_data_right, tracking_data_left, homography_matrix_left, homography_matrix_right,
    dimensions_file, output_right_json, output_right_non_json, output_left_json, output_left_non_json
):
    global TRACKING_DATA_RIGHT, TRACKING_DATA_LEFT, HOMOGRAPHY_MATRIX_LEFT, HOMOGRAPHY_MATRIX_RIGHT
    global DIMENSIONS_FILE, OUTPUT_RIGHT_JSON, OUTPUT_RIGHT_NON_INTERSECTIONS_JSON
    global OUTPUT_LEFT_JSON, OUTPUT_LEFT_NON_INTERSECTIONS_JSON, DEBUG_IMAGE_RIGHT, DEBUG_IMAGE_LEFT

    TRACKING_DATA_RIGHT = tracking_data_right
    TRACKING_DATA_LEFT = tracking_data_left
    HOMOGRAPHY_MATRIX_LEFT = homography_matrix_left
    HOMOGRAPHY_MATRIX_RIGHT = homography_matrix_right
    DIMENSIONS_FILE = dimensions_file
    OUTPUT_RIGHT_JSON = output_right_json
    OUTPUT_RIGHT_NON_INTERSECTIONS_JSON = output_right_non_json
    OUTPUT_LEFT_JSON = output_left_json
    OUTPUT_LEFT_NON_INTERSECTIONS_JSON = output_left_non_json

    main()
