import cv2
import json
import numpy as np

# File paths
INPUT_VIDEO = "output.mp4"
JSON_INPUT = "output.json"
POINTS_JSON = "points.json"
OUTPUT_VIDEO = "processed_output.mp4"
OUTPUT_JSON = "processed_output.json"

color_map = {
    "blue":   (255, 0, 0),
    "purple": (255, 0, 255),
    "red":    (0, 0, 255),
    "orange": (0, 165, 255),
    "yellow": (0, 255, 255)
}

with open(JSON_INPUT, "r") as f:
    data = json.load(f)

with open(POINTS_JSON, "r") as f:
    points = json.load(f)

dst_points = np.array([
    points["top_left"],
    points["top_right"],
    points["bottom_right"],
    points["bottom_left"]
], dtype=np.float32)

src_points = np.array([
    [0, 0],
    [10, 0],
    [10, 300],
    [0, 300]
], dtype=np.float32)

H = cv2.getPerspectiveTransform(src_points, dst_points)

cap = cv2.VideoCapture(INPUT_VIDEO)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

LEFT_VIDEO_WIDTH = 750

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (frame_width, frame_height))

frame_idx = 0
json_output = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret or frame_idx >= len(data["frames"]):
        break

    frame_data = data["frames"][frame_idx]
    saved_objects = []

    for obj in frame_data["obj"]:
        color = obj["color"]
        t_c = obj["t_c"]
        bbox = obj["bbox"]
        x1, y1, x2, y2 = map(int, bbox)

        draw_color = color_map.get(color, (255, 255, 255))
        font_scale = 0.3
        thickness = 1

        # Track bbox to save
        bbox_to_save = [x1, y1, x2, y2]
        subtract_y_for_json = False
        save_it = False

        if color in ["blue"]:
            save_it = True
            cv2.rectangle(frame, (x1, y1), (x2, y2), draw_color, 2)
            cv2.putText(frame, str(t_c[0]), (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale,  color_map.get("blue", (255, 255, 255)), thickness)
        elif color in ["red"]:
            save_it = True

            offset_x = LEFT_VIDEO_WIDTH
            cv2.rectangle(frame, (x1 + offset_x, y1), (x2 + offset_x, y2), draw_color, 2)
            cv2.putText(frame, str(t_c[0]), (x1 + offset_x, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_map.get("red", (255, 255, 255)), thickness)
            subtract_y_for_json = True  # subtract y1/y2 by 10 in JSON

        elif color in ["yellow", "orange"]:
            if 25 < t_c[0] < 35:
                save_it = True

                src_x = t_c[0] - 30
                
                src_point = np.array([[[src_x, 0]]], dtype=np.float32)
                dst_point = cv2.perspectiveTransform(src_point, H)
                converted_center = dst_point[0][0]

                y_center_bbox = (y1 + y2) / 2

                if (color in ["yellow", "orange"]) and (converted_center[0] < frame_width / 2):
                    y_center_bbox -= 0
                else:
                    subtract_y_for_json = False #maybe change this

                width = x2 - x1
                height = y2 - y1

                new_y1 = int(y_center_bbox - height / 2)
                new_y2 = int(y_center_bbox + height / 2)

                new_x2 = int(converted_center[0] - width / 2)
                new_x1 = int(converted_center[0] + width / 2)
                    
                
                cv2.rectangle(frame, (new_x1, new_y1), (new_x2, new_y2), color_map.get(color, (255, 255, 255)), 2)
                cv2.putText(frame, str(t_c[0]), (new_x1, new_y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, draw_color, thickness)
                
                
                bbox_to_save = [new_x1, new_y1, new_x2, new_y2]

        # Apply y subtraction for JSON only
        if subtract_y_for_json:
            bbox_to_save[1] -= 0  # y1
            bbox_to_save[3] -= 0  # y2
            bbox_to_save[0] += LEFT_VIDEO_WIDTH
            bbox_to_save[2] += LEFT_VIDEO_WIDTH

        # Offset t_c[0] by +345 for red/orange/yellow, keep t_c[1] unchanged

        if color in ["red", "orange", "yellow"]:
            saved_tc = [float(t_c[0]) + 340, float(t_c[1])]
        else:
            saved_tc = [float(t_c[0]), float(t_c[1])]

        if save_it:
                saved_objects.append({
            "bbox": [float(v) for v in bbox_to_save],
            "t_c": saved_tc,
            "color": color
        })

        

    json_output.append({
        "objects": saved_objects
    })

    out.write(frame)
    frame_idx += 1

cap.release()
out.release()

# Save JSON output
with open(OUTPUT_JSON, "w") as f:
    json.dump({"frames": json_output}, f, indent=2)

print(f"Processed video saved as {OUTPUT_VIDEO}")
print(f"Processed JSON saved as {OUTPUT_JSON}")
