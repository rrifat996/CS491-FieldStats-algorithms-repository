import cv2
import json
from tqdm import tqdm

# ▶️ Configuration
INPUT_VIDEO   = 'input.mp4'
JSON_FILE     = 'converted_output3.json'
OUTPUT_VIDEO  = 'output2.mp4'
FRAME_SIZE    = (800, 300)       # (width, height)
FOURCC        = cv2.VideoWriter_fourcc(*'mp4v')
BOX_THICKNESS = 2
CENTER_RADIUS = 3
MAX_FRAMES    = 1000

# simple color map (BGR)
COLOR_MAP = {
    'blue':   (255,   0,   0),
    'red':    (  0,   0, 255),
    'green':  (  0, 255,   0),
    'yellow': (  0, 255, 255),
}

# offset map
COLOR_OFFSETS = {
    'blue':   (0,   0),
    'red':    (0, 0),
    'orange': (0, 0),
    'yellow': (0, 0),
}


def load_annotations(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    by_frame = {}
    for idx, frame in enumerate(data.get('frames', [])):
        by_frame[idx] = frame.get('objects', [])
    return by_frame


def main():
    ann = load_annotations(JSON_FILE)

    cap = cv2.VideoCapture(INPUT_VIDEO)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {INPUT_VIDEO}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    out = cv2.VideoWriter(OUTPUT_VIDEO, FOURCC, fps, FRAME_SIZE)

    for frame_idx in tqdm(range(MAX_FRAMES), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, FRAME_SIZE)

        for obj in ann.get(frame_idx, []):
            color_name = obj.get('color', '').lower()

            if color_name == 'yellow':
                tc = obj.get('t_c')
                if tc is None or not (25 < tc[0] < 35):
                    continue

            x1, y1, x2, y2 = map(int, obj['bbox'])
            col = COLOR_MAP.get(color_name, (255, 255, 255))
            dx, dy = COLOR_OFFSETS.get(color_name, (0, 0))

            x1_off, y1_off = x1 + dx, y1 + dy
            x2_off, y2_off = x2 + dx, y2 + dy
            cv2.rectangle(frame, (x1_off, y1_off), (x2_off, y2_off), col, BOX_THICKNESS)

            if 't_c' in obj:
                cx, cy = map(int, obj['t_c'])
                cx_off, cy_off = cx + dx, cy + dy
                cv2.circle(frame, (cx_off, cy_off), CENTER_RADIUS, col, -1)

        out.write(frame)

    cap.release()
    out.release()
    print(f"Written annotated video to {OUTPUT_VIDEO}")


if __name__ == '__main__':
    main()
