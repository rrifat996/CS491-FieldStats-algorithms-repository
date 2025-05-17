#!/usr/bin/env python3
"""
extract_dots.py

Given a tracking file (the ByteTrack-style structure) and a
reference frame interval X, this script collects object centres for every
multiple of X (e.g., X, 2X, 3X, ...) at the following frames:

    • frame  R-10   → colour "red"
    • frame  R      → colour "yellow"
    • frame  R+10   → colour "blue"

If no arguments are provided, defaults are:
    --track-file output.json
    --frame      100
    --out        dots_100.json

Output is a JSON array of objects:
[
  {
    "reference_frame": R,
    "dots": [ ... ]
  },
  ...
]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Map frame offsets to output colours
ColourMap = {0: "red", 10: "yellow", 20: "blue"}  # offsets → colour


def load_tracks(path: Path) -> Dict[int, List[dict]]:
    """
    Convert `tracks` section into {frame_no: [objects …]} for fast lookup.
    """
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    frame_dict: Dict[int, List[dict]] = {}
    for frame in data.get("tracks", []):
        frame_dict[frame["fr"]] = frame["obj"]
    return frame_dict


def centre_of(obj: dict) -> Tuple[float, float]:
    """
    Prefer the pre-computed 'c' centre; fall back to bbox mid-point otherwise.
    """
    if "c" in obj and isinstance(obj.get("c"), list) and len(obj["c"]) == 2:
        return float(obj["c"][0]), float(obj["c"][1])

    x1, y1, x2, y2 = obj.get("bbox", [0, 0, 0, 0])
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def collect_dots(frames: Dict[int, List[dict]], R: int) -> List[dict]:
    """
    Gather centres, original bounding boxes, and colours for frames {R-10, R, R+10}.
    """
    dots: List[dict] = []
    for offset, colour in ColourMap.items():
        fr_no = R - 10 + offset  # 0→R-10, 10→R, 20→R+10
        objs = frames.get(fr_no)
        if objs is None:
            print(f"[warning] Frame {fr_no} not found – skipping.", file=sys.stderr)
            continue

        for obj in objs:
            cx, cy = centre_of(obj)
            dots.append({
                "x": cx,
                "y": cy,
                "frame": fr_no,
                "id": obj.get("id"),
                "cls_id": obj.get("cls_id"),
                "color": colour,
                "bbox": obj.get("bbox"),
                "t_c": obj.get("t_c"),
            })
    return dots


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract colour-coded centres from tracking JSON at every multiple of X."
    )
    ap.add_argument(
        "--track-file", type=Path, default=Path("output.json"),
        help="Input tracking JSON (default: output.json)"
    )
    ap.add_argument(
        "--frame", type=int, default=100,
        help="Reference frame interval X (default: 10)"
    )
    ap.add_argument(
        "--out", type=Path, default=Path("dots_100.json"),
        help="Output JSON file (default: dots_100.json)"
    )
    args = ap.parse_args()

    frames = load_tracks(args.track_file)
    if not frames:
        print("[error] No frames found in tracking file.", file=sys.stderr)
        sys.exit(1)

    max_frame = max(frames.keys())
    interval = args.frame
    results = []

    for ref in range(interval, max_frame + 1, interval):
        dots = collect_dots(frames, ref)
        results.append({
            "reference_frame": ref,
            "dots": dots,
        })

    with args.out.open("w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)

    total_refs = len(results)
    total_dots = sum(len(r["dots"]) for r in results)
    print(f"[✓] Wrote {total_dots} dots across {total_refs} reference frames to {args.out}")


if __name__ == "__main__":
    main()