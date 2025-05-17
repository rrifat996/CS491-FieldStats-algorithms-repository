import cv2
import json
import os

# --- parameters ---
VIDEO_IN    = "right5.mp4"
VIDEO_OUT   = "out2.mp4"
JSON_IN     = "80_final.json"
MAX_FRAMES  = 500  # only process this many frames

# map CSS‐style color names to BGR tuples
COLOR_MAP = {
    "red":    (0,   0,   255),
    "green":  (0,   255, 0),
    "blue":   (255, 0,   0),
    "orange": (0,   165, 255),
    "purple": (255, 0,   255),
    "yellow": (0,   255, 255),
}

# --- check file write permissions ---
if os.path.exists(VIDEO_OUT):
    print(f"[DEBUG] Output file {VIDEO_OUT} exists. Attempting to overwrite.")
    if not os.access(VIDEO_OUT, os.W_OK):
        raise PermissionError(f"[ERROR] Cannot write to {VIDEO_OUT} — check file permissions.")
else:
    print(f"[DEBUG] Output file {VIDEO_OUT} does not exist yet. Creating new file.")

# --- load JSON annotations ---
print(f"[DEBUG] Loading annotations from {JSON_IN}...")
with open(JSON_IN, "r") as f:
    frames_data = json.load(f)

print(f"[DEBUG] Loaded annotations for {len(frames_data)} frames.")

# build quick lookup: frame_index → list of objects
by_frame = { item["frame_index"]: item["objects"] for item in frames_data }

# --- open input video ---
print(f"[DEBUG] Opening video {VIDEO_IN}...")
cap = cv2.VideoCapture(VIDEO_IN)
if not cap.isOpened():
    raise RuntimeError(f"[ERROR] Cannot open video {VIDEO_IN}")
else:
    print("[DEBUG] Input video opened successfully.")

fps    = cap.get(cv2.CAP_PROP_FPS)
w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"[DEBUG] Input video FPS: {fps}, Resolution: {w}x{h}")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out    = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (w, h))

if not out.isOpened():
    raise RuntimeError(f"[ERROR] Cannot open video writer for {VIDEO_OUT}")
else:
    print("[DEBUG] Output video writer initialized successfully.")

# --- process frames ---
frame_idx = 0
while frame_idx < MAX_FRAMES:
    ret, frame = cap.read()
    if not ret:
        break

    objs = by_frame.get(frame_idx, [])

    for obj in objs:
        x1, y1, x2, y2 = map(int, obj["bbox"])
        colname        = obj.get("color", "").lower()
        color          = COLOR_MAP.get(colname, (255, 255, 255))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Add debug watermark to ensure visual change
    cv2.putText(frame, f"Frame {frame_idx}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    out.write(frame)
    frame_idx += 1

# --- cleanup ---
cap.release()
out.release()
print(f"[DEBUG] Released video resources.")
print(f"[INFO] Done — saved first {frame_idx} frames with bboxes to {VIDEO_OUT}")
