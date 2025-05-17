#!/usr/bin/env python3
"""
run_team34_on_boxes.py   ·  v4  (multi-frame, default I/O)

Reads <video>.mp4 and <boxes>.json—which may be a list of
{reference_frame, bboxes} entries—and for each reference frame:

  • Removes the two goalkeepers (left-most/right-most by t_c[0])
    so exactly 21 crops remain
  • Extracts crops, classifies 21 images via team34.py,
  • Assigns teams (including GKs) and prints GK t_c[0]

Writes all results to the output JSON as an array of entries,
each with reference_frame and updated bboxes.

Defaults (if no flags provided):
    --video output.mp4
    --boxes selected_bboxes.json
    --output output.json

Usage:
    python run_team34_on_boxes.py
    python run_team34_on_boxes.py --video my.mp4 --boxes my_boxes.json --output my_out.json
"""
import argparse
import cv2
import importlib
import json
import numpy as np
from pathlib import Path
import shutil
from scipy.optimize import linear_sum_assignment
import sys

# ------------------------------------------------------------
# utilities
# ------------------------------------------------------------
def goalkeeper_indices(bboxes):
    coor = [bb["t_c"][0] for bb in bboxes]
    idx_left  = int(np.argmin(coor))
    idx_right = int(np.argmax(coor))
    return idx_left, idx_right


def extract_crops(video_path, bboxes, img_dir, keep_indices):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")
    by_frame = {}
    for orig_idx in keep_indices:
        item = bboxes[orig_idx]
        by_frame.setdefault(item["frame"], []).append((orig_idx, item["bbox"]))
    result = []
    for frame_no, lst in by_frame.items():
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Could not read frame {frame_no}")
        h, w = frame.shape[:2]
        for orig_idx, (x1, y1, x2, y2) in lst:
            x1, y1 = max(int(x1), 0), max(int(y1), 0)
            x2, y2 = min(int(x2), w-1), min(int(y2), h-1)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                raise ValueError(f"Empty crop for bbox index {orig_idx}")
            out_path = img_dir / f"{orig_idx:02d}.jpg"
            cv2.imwrite(str(out_path), crop)
            result.append((orig_idx, out_path))
    cap.release()
    result.sort(key=lambda tup: tup[0])
    return result


def load_entries(path: Path):
    if not path.exists():
        sys.exit(f"Error: '{path}' not found.")
    with path.open('r', encoding='utf-8') as fp:
        data = json.load(fp)
    if isinstance(data, dict) and 'bboxes' in data and 'reference_frame' in data:
        return [data]
    if isinstance(data, list):
        return data
    sys.exit("Error: input JSON must be dict or list with 'reference_frame' and 'bboxes'.")


def process_entry(entry, video_path):
    data = entry.copy()
    bboxes = data.get('bboxes', [])
    if len(bboxes) > 21:
        gk_left, gk_right = goalkeeper_indices(bboxes)
        field_indices = [i for i in range(len(bboxes)) if i not in (gk_left, gk_right)]
    else:
        gk_left = gk_right = None
        field_indices = list(range(len(bboxes)))
    if gk_left is not None:
        print(f"Reference {data['reference_frame']}: Left GK t_c[0]={bboxes[gk_left]['t_c'][0]}, Right GK t_c[0]={bboxes[gk_right]['t_c'][0]}")
    img_dir = Path('images')
    if img_dir.exists(): shutil.rmtree(img_dir)
    img_dir.mkdir()
    crops = extract_crops(video_path, bboxes, img_dir, field_indices)
    team34 = importlib.import_module('team34')
    t1_col = np.array(team34.team1_color)
    t2_col = np.array(team34.team2_color)
    t3_col = np.array(team34.team3_color)
    images_info = []
    for orig_idx, p in crops:
        img = cv2.imread(str(p))
        dom, _, _ = team34.extract_player_info(img, debug=False)
        images_info.append({'orig_idx': orig_idx, 'dom': dom})
    dists3 = [np.linalg.norm(info['dom']-t3_col) for info in images_info]
    ref_idx = int(np.argmin(dists3))
    images_info[ref_idx]['team'] = 3
    remaining = [inf for inf in images_info if 'team' not in inf]
    cost = np.zeros((len(remaining), len(remaining)))
    half = len(remaining)//2
    for r, inf in enumerate(remaining):
        d1 = np.linalg.norm(inf['dom']-t1_col)
        d2 = np.linalg.norm(inf['dom']-t2_col)
        cost[r, :half] = d1
        cost[r, half:] = d2
    rows, cols = linear_sum_assignment(cost)
    for r, c in zip(rows, cols):
        remaining[r]['team'] = 1 if c < half else 2
    for info in images_info:
        bboxes[info['orig_idx']]['team'] = info['team']
    if gk_left is not None:
        bboxes[gk_left]['team'] = 2
        bboxes[gk_right]['team'] = 1
    return {'reference_frame': data['reference_frame'], 'bboxes': bboxes}


def main():
    ap = argparse.ArgumentParser(description='Classify bboxes for each reference frame')
    ap.add_argument('--video', default='output_team.mp4', help='Video file (default: output.mp4)')
    ap.add_argument('--boxes', default='selected_bboxes.json', help='Input JSON (default: selected_bboxes.json)')
    ap.add_argument('--output', default='output.json', help='Output JSON (default: output.json)')
    args = ap.parse_args()
    entries = load_entries(Path(args.boxes))
    results = []
    for entry in entries:
        results.append(process_entry(entry, Path(args.video)))
    with open(args.output, 'w', encoding='utf-8') as fp:
        json.dump(results, fp, indent=2)
    print(f"✓ Done: processed {len(results)} frames → {args.output}")

if __name__ == '__main__':
    main()
