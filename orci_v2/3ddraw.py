#!/usr/bin/env python3
"""
annotate_video.py

Draws the tracked bounding boxes from final_result.json onto output.mp4,
using the same per‐team colors and ID labels as Mode H.
"""
import json
import cv2
from pathlib import Path

# ───────────────────── constants ─────────────────────

INPUT_VIDEO  = "output.mp4"
INPUT_JSON   = "final_result.json"
OUTPUT_VIDEO = "output_annotated.mp4"

# Team → BGR colour mapping
TEAM_COLORS = {
    1: (0, 255,   0),   # green
    2: (255,255, 255),  # white
    3: (0, 255, 255),   # yellow
}

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
THICKNESS = 2

# ───────────────────── helpers ─────────────────────

def load_tracks_by_frame(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    by_frame = {}
    for entry in data.get("tracks", []):
        fr = entry.get("fr")
        objs = entry.get("obj", [])
        by_frame[fr] = objs
    return by_frame

def draw_team_boxes(frame, objs):
    """
    Draw each bbox in its team's colour, and label with its ID.
    """
    for o in objs:
        bbox = o.get("bbox", [0,0,0,0])
        x1, y1, x2, y2 = map(int, bbox)
        team = o.get("team", 0)
        col = TEAM_COLORS.get(team, (0,0,255))  # default red for unknown
        # draw rectangle
        cv2.rectangle(frame, (x1,y1), (x2,y2), col, THICKNESS)
        # draw ID label
        id_lbl = o.get("id")
        if id_lbl is not None:
            text = str(id_lbl)
            cv2.putText(frame, text, (x1, y1 - 5),
                        FONT, FONT_SCALE, col, THICKNESS)

# ───────────────────── main ─────────────────────

def main():
    tracks = load_tracks_by_frame(INPUT_JSON)

    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        print(f"❌ Cannot open video {INPUT_VIDEO}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for frame_idx in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        objs = tracks.get(frame_idx, [])
        if objs:
            draw_team_boxes(frame, objs)

        writer.write(frame)

    cap.release()
    writer.release()
    print(f"✓ Wrote annotated video → {OUTPUT_VIDEO}")

if __name__ == "__main__":
    main()
