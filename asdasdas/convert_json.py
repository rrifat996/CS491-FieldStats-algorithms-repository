import json

# Default values to fill missing fields
DEFAULT_OBJECT = {
    "class_id": 0,
    "confidence": 0.0,
    "color": "blue",
    "source": "left",
    "center": [0.0, 0.0],
    "team_index": 1,
    "team_classification_confidence": 0.99
}

def convert_json(input_data):
    output_data = {"frames": []}

    for idx, frame in enumerate(input_data.get("frames", [])):
        new_frame = {"objects": []}
        new_frame["frame_index"] = frame.get("frame_index", idx)

        for obj in frame.get("objects", []):
            new_obj = DEFAULT_OBJECT.copy()
            new_obj.update(obj)  # Merge existing keys

            # Handle t_c preservation
            if "t_c" in obj:
                new_obj["t_c"] = obj["t_c"]
            elif "transformed_center" in obj:
                new_obj["t_c"] = obj["transformed_center"]

            new_frame["objects"].append(new_obj)

        output_data["frames"].append(new_frame)

    return output_data

# Example usage
with open("processed_output.json", "r") as infile:
    input_json = json.load(infile)

converted = convert_json(input_json)

with open("converted_output3.json", "w") as outfile:
    json.dump(converted, outfile, indent=4)

print("Conversion complete.")
