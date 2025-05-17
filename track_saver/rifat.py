import os
import json
import cv2
import supervision as sv  # Includes ByteTrack implementation
import numpy as np

# ==== SET THESE VARIABLES ====
video_filename = "output.mp4"
bbox_json_filename = "processed_output.json"
output_video_filename = "tracked_output.mp4"

def load_bboxes_from_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data["frames"]


def track_and_draw(video_path, bbox_json_path, output_video_path):
    frames_data = load_bboxes_from_json(bbox_json_path)
    tracker = sv.ByteTrack()

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (w, h))

    for idx, frame_info in enumerate(frames_data):
        ret, frame = cap.read()
        if not ret:
            break

        # Extract bboxes from the frame
        bboxes = [obj["bbox"] for obj in frame_info["objects"]]

        if len(bboxes) > 0:
            bboxes_np = np.array(bboxes, dtype=np.float32)
            detections = sv.Detections(
                xyxy=bboxes_np,
                confidence=np.ones(len(bboxes)),
                class_id=np.zeros(len(bboxes), dtype=int)
            )

            tracks = tracker.update_with_detections(detections)

            for xyxy, track_id in zip(tracks.xyxy, tracks.tracker_id):
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 1)
                cv2.putText(frame, f'ID {track_id}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Output video saved to {output_video_path}")


# Run tracking
track_and_draw(video_filename, bbox_json_filename, output_video_filename)