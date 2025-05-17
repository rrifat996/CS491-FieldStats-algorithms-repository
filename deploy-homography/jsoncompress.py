import json
from tqdm import tqdm  # Import tqdm for progress bars

def round_floats(obj):
    """
    Recursively round every float in the object to one decimal place.
    If the object is a list, process each item.
    If the object is a dict, process each value.
    Otherwise, return the object as is.
    """
    if isinstance(obj, float):
        return round(obj, 1)
    elif isinstance(obj, list):
        return [round_floats(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: round_floats(value) for key, value in obj.items()}
    else:
        return obj

def main():
    input_file = "80_final.json"
    output_file = "80_iou_compressed.json"

    # Read the original JSON file
    with open(input_file, "r") as f:
        data = json.load(f)
    
    # Check if the JSON is a list (frames only) or a dict with metadata and frames.
    if isinstance(data, list):
        frames = data
        metadata = {}
        new_data = {"metadata": metadata, "frames": frames}
    elif isinstance(data, dict):
        metadata = data.get("metadata", {})
        frames = data.get("frames", [])
        new_data = {"metadata": metadata, "frames": frames}
    else:
        print("Unexpected JSON structure.")
        return

    # Process each frame with tqdm progress bar
    for frame in tqdm(new_data["frames"], desc="Processing Frames"):
        # Rename 'frame_index' to 'fr'
        if "frame_index" in frame:
            frame["fr"] = frame.pop("frame_index")
        
        # Rename 'objects' to 'obj'
        if "objects" in frame:
            frame["obj"] = frame.pop("objects")
        
        # Process each object in the frame with tqdm progress bar
        for obj in frame.get("obj", []):
            # Remove unwanted keys (but keep 'color')
            for key in ["class_id", "confidence", "center"]:
                obj.pop(key, None)
            
            # Remove the 'source' key (no 'src' is added)
            if "source" in obj:
                obj.pop("source")
            
            # Rename 'transformed_center' to 't_c'
            if "transformed_center" in obj:
                obj["t_c"] = obj.pop("transformed_center")

    # Recursively round all floating-point numbers to one decimal place.
    new_data = round_floats(new_data)

    # Write the modified JSON data to the output file.
    with open(output_file, "w") as f:
        json.dump(new_data, f, indent=4)

    print("JSON conversion complete. New file saved as", output_file)

    print("eighth script completed. ALL COMPLETED!!!!!!")

if __name__ == "__main__":
    main()
