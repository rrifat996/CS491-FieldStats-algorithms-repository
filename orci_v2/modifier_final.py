import json

# 1. Load the data
with open("result.json", "r") as f:
    data = json.load(f)

# 2. Filter out objs with bbox x‑difference exactly 5
for track in data.get("tracks", []):
    filtered_objs = []
    for obj in track.get("obj", []):
        x1, _, x2, _ = obj["bbox"]
        center_x = (x1 + x2) / 2
        if abs(x2 - x1) != 5 and center_x >= 50:
            filtered_objs.append(obj)
    track["obj"] = filtered_objs

# 3. Write the cleaned JSON
with open("result2.json", "w") as f:
    json.dump(data, f, indent=4)
