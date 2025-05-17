import json
import itertools
from tqdm import tqdm


def calculate_iou(bbox1, bbox2):
    # Calculate intersection
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate union
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    union = area1 + area2 - intersection

    return intersection / union if union != 0 else 0

def remove_low_conf_objects(json_file, output_file, iou_threshold=0.80):
    with open(json_file, 'r') as file:
        data = json.load(file)

    total_objects_before = sum(len(frame['objects']) for frame in data)

    group1_colors = {"yellow", "red", "orange"}
    group2_colors = {"blue", "purple", "pink"}

    for frame in tqdm(data, desc="Processing frames", unit="frame"):

        objects = frame['objects']
        to_remove = set()

        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects):
                if i >= j:  # Avoid duplicate comparisons
                    continue

                # Skip comparison if objects belong to different color groups
                if (obj1['color'] in group1_colors and obj2['color'] in group2_colors) or \
                   (obj1['color'] in group2_colors and obj2['color'] in group1_colors):
                    continue

                iou = calculate_iou(obj1['bbox'], obj2['bbox'])
                if iou > iou_threshold:
                    if obj1['confidence'] < obj2['confidence']:
                        to_remove.add(i)
                    else:
                        to_remove.add(j)

        frame['objects'] = [obj for idx, obj in enumerate(objects) if idx not in to_remove]

    total_objects_after = sum(len(frame['objects']) for frame in data)

    with open(output_file, 'w') as file:
        json.dump(data, file, indent=4)

    print(f"Total objects before: {total_objects_before}")
    print(f"Total objects after: {total_objects_after}")

def main():
    input_file = "borderfiltered_merged_output_with_transformed_center.json"  # Replace with the path to your JSON file
    output_file = "80_final.json"  # Replace with the desired output file name

    remove_low_conf_objects(input_file, output_file)

    import jsoncompress
    print("seventh script completed. Now running the eighth script...")
    jsoncompress.main()  # Call the second script after finishing the first

if __name__ == "__main__":
    input_file = "borderfiltered_merged_output_with_transformed_center.json"  # Replace with the path to your JSON file
    output_file = "80_final.json"  # Replace with the desired output file name

    remove_low_conf_objects(input_file, output_file)

