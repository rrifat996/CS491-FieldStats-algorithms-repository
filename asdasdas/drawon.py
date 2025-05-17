import cv2
import json

# Config
VIDEO_PATH = "processed_output.mp4"
JSON_PATH = "result.json"
OUTPUT_PATH = "bbox_output1.mp4"
BOX_SIZE = 10  # 10x10 box

# Load bbox data
with open(JSON_PATH, "r") as f:
    data = json.load(f)

# Load video
cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Output writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

# Convert bbox data to dict for fast lookup: {frame_index: [obj_list]}
frame_to_objects = {track["fr"]: track["obj"] for track in data["tracks"]}

frame_index = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw boxes for current frame if available
    if frame_index in frame_to_objects:
        for obj in frame_to_objects[frame_index]:
            x, y = obj["c"]
            top_left = (int(x - BOX_SIZE / 2), int(y - BOX_SIZE / 2))
            bottom_right = (int(x + BOX_SIZE / 2), int(y + BOX_SIZE / 2))

            # Draw rectangle and ID text
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 1)
            cv2.putText(frame, f'ID:{obj["id"]}', (int(x) + 6, int(y) - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    out.write(frame)
    frame_index += 1

# Cleanup
cap.release()
out.release()
cv2.destroyAllWindows()

print("Done drawing bounding boxes.")
