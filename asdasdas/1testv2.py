#!/usr/bin/env python3
"""
merge_videos.py

Merge two videos side by side with resizing and vertical shift for the right video,
show a progress bar using tqdm, then save the combined output.

Defaults:
  left input:  l5.mp4
  right input: r5.mp4
  output:      output.mp4
  dimensions:  WIDTH=750, HEIGHT=500
  shift up:    SHIFT_UP=10 pixels

Usage:
  python merge_videos.py [--left LEFT] [--right RIGHT] [--output OUT]
                         [--width W] [--height H] [--shift S]
"""
import argparse
import cv2
import numpy as np
import sys
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser(description="Merge two videos side by side with resizing, shift, and progress bar")
    ap.add_argument('--left',   default='l5.mp4', help='Left video path')
    ap.add_argument('--right',  default='r5.mp4', help='Right video path')
    ap.add_argument('--output', default='output.mp4', help='Output video path')
    ap.add_argument('--width',  type=int, default=750,    help='Frame width per video')
    ap.add_argument('--height', type=int, default=500,    help='Frame height per video')
    ap.add_argument('--shift',  type=int, default=10,     help='Pixels to shift the right video upward')
    args = ap.parse_args()

    # Open video captures
    cap_l = cv2.VideoCapture(args.left)
    cap_r = cv2.VideoCapture(args.right)
    if not cap_l.isOpened() or not cap_r.isOpened():
        sys.exit("Error: cannot open input videos.")

    # Retrieve total frame count for progress bar
    total_frames = int(cap_l.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap_l.get(cv2.CAP_PROP_FPS)

    out_width, out_height = args.width * 2, args.height
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (out_width, out_height))

    # Process frames with tqdm progress bar
    for _ in tqdm(range(total_frames), desc="Merging frames"):
        ret1, f1 = cap_l.read()
        ret2, f2 = cap_r.read()
        if not ret1 or not ret2:
            break

        # Resize frames
        f1 = cv2.resize(f1, (args.width, args.height))
        f2 = cv2.resize(f2, (args.width, args.height))

        # Shift upward f2
        blank = np.zeros_like(f2)
        s = args.shift
        if 0 <= s < args.height:
            blank[0:args.height-s, :] = f2[s:args.height, :]
        else:
            blank[:] = f2

        # Combine horizontally and write
        combined = np.hstack((f1, blank))
        out.write(combined)

    # Release resources
    cap_l.release()
    cap_r.release()
    out.release()

    print(f"Done. Merged video saved to {args.output}")


if __name__ == '__main__':
    main()
