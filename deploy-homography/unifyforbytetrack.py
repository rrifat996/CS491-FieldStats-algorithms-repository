import json
import numpy as np
import os
import cv2
from tqdm import tqdm

# TODO LEFT RIGHT COLORA GORE OLACAK
# Input JSON files
JSON_FILES = {
    "new_left_intersections.json": "left",
    "left_non_intersections.json": "left",
    "new_right_intersections.json": "right",
    "right_non_intersections.json": "right"
}

# Homography matrices
HOMOGRAPHY_MATRIX_LEFT = "al2_homography_matrix.txt"
HOMOGRAPHY_MATRIX_RIGHT = "al1_homography_matrix.txt"

# Output JSON file
OUTPUT_JSON = "merged_output_with_transformed_center.json"


def load_homography_matrices():
    """
    Load homography matrices from file.
    """
    homography_matrix_left = np.loadtxt(HOMOGRAPHY_MATRIX_LEFT, delimiter=' ')
    homography_matrix_right = np.loadtxt(HOMOGRAPHY_MATRIX_RIGHT, delimiter=' ')
    return homography_matrix_left, homography_matrix_right


def transform_point(point, homography_matrix):
    """
    Transform a single point using a homography matrix.
    """
    point = np.array([[point]], dtype=np.float32)
    transformed_point = cv2.perspectiveTransform(point, homography_matrix)
    return transformed_point[0][0].tolist()  # Convert to a list for JSON compatibility


def merge_jsons(json_files, homography_matrix_left, homography_matrix_right):
    """
    Merge multiple JSON files, grouping objects by frame_index and adding a source field and transformed_center.
    Apply filtering based on transformed center coordinates.
    """
    merged_data = {}

    for json_file, source in tqdm(json_files.items(), desc="Processing JSON files"):
        if not os.path.exists(json_file):
            print(f"Error: File '{json_file}' not found.")
            continue

        with open(json_file, "r") as f:
            data = json.load(f)

        for frame in tqdm(data, desc=f"Processing frames in {json_file}", leave=False):
            frame_index = frame["frame_index"]
            objects = frame.get("objects", [])

            filtered_objects = []
            for obj in objects:


                # Add source field
                obj["source"] = source

                # Compute transformed center using the appropriate homography matrix
                homography_matrix = homography_matrix_left if source == "left" else homography_matrix_right

                bbox = obj["bbox"]
                bottom_middle = [
                    (bbox[0] + bbox[2]) / 2,  # x: middle between left and right
                    bbox[3]                  # y: bottom of the box
                ]

                obj["transformed_center"] = transform_point(bottom_middle, homography_matrix)

                # Apply filtering
                transformed_center_x = obj["transformed_center"][0]

                
                if source == "left" and transformed_center_x > 370:
                    continue  # Skip this object
                if source == "right" and transformed_center_x < 30:
                    continue  # Skip this object

                filtered_objects.append(obj)

            if frame_index not in merged_data:
                merged_data[frame_index] = {"frame_index": frame_index, "objects": []}

            # Append filtered objects to the appropriate frame
            merged_data[frame_index]["objects"].extend(filtered_objects)

    # Convert merged_data to a list sorted by frame_index
    return sorted(merged_data.values(), key=lambda x: x["frame_index"])


def save_json(data, output_file):
    """
    Save the merged JSON data to a file.
    """
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Merged JSON saved as '{output_file}'")


def main():
    # Load homography matrices
    homography_matrix_left, homography_matrix_right = load_homography_matrices()

    # Merge the JSON files
    merged_data = merge_jsons(JSON_FILES, homography_matrix_left, homography_matrix_right)

    # Save the merged data to a new JSON file
    save_json(merged_data, OUTPUT_JSON)


    import filterjson3
    print("fifth script completed. Now running the sixth script...")
    filterjson3.main()  # Call the second script after finishing the first


if __name__ == "__main__":
    main()
    
