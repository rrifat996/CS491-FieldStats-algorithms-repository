#!/usr/bin/env python3
"""
multi_viewer.py  –  video viewer with up to eight detection modes:

Modes
-----
  A: ByteTrack bounding-boxes
  B: second bbox format
  C: plain dots (all frames)
  D: clustered dots
  E: selected boxes (per frame)
  F: selected boxes (all frames)
  G: team-colored selected boxes (per frame only; warning otherwise)
  H: final_result mappings (per-frame) from final_result.json

Hotkeys
--------
  ←/a    previous frame
  →/d    next frame
  z      jump to previous multiple of 100
  c      jump to next multiple of 100
  g      go to arbitrary frame via console input
  m      cycle mode
  SPACE  play/pause
  s      save current frame to PNG
  q      quit
"""
import argparse
import json
import sys
from pathlib import Path

import cv2

# ───────────────────── loaders ─────────────────────

def load_bytetrack(path: str) -> dict[int, list[dict]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {fr["fr"]: fr["obj"] for fr in data.get("tracks", [])}

def load_boxes(path: str) -> dict[int, list[dict]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        i: frame.get("objects", [])
        for i, frame in enumerate(data.get("frames", []))
    }

def load_dots(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        dots = []
        for entry in data:
            dots.extend(entry.get("dots", []))
        return dots
    else:
        return data.get("dots", [])

def load_clustered_dots(path: str) -> dict[int, list[dict]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    all_dots = []
    if isinstance(data, list):
        for entry in data:
            all_dots.extend(entry.get("dots", []))
    else:
        all_dots = data.get("dots", [])
    clusters: dict[int, list[dict]] = {}
    for d in all_dots:
        cid = d.get("cluster", 0)
        clusters.setdefault(cid, []).append(d)
    return clusters

def load_selected_boxes(path: str) -> dict[int, list[dict]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    by_frame: dict[int, list[dict]] = {}
    if isinstance(data, list):
        for entry in data:
            fr = entry.get("reference_frame")
            for b in entry.get("bboxes", []):
                by_frame.setdefault(fr, []).append(b)
    else:
        for b in data.get("bboxes", []):
            fr = b.get("frame")
            by_frame.setdefault(fr, []).append(b)
    return by_frame

def load_team_boxes_by_frame(path: str) -> dict[int, list[dict]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    by_frame: dict[int, list[dict]] = {}
    if isinstance(data, list):
        for entry in data:
            fr = entry.get("reference_frame")
            by_frame[fr] = entry.get("bboxes", [])
    else:
        for b in data.get("bboxes", []):
            fr = b.get("frame")
            by_frame.setdefault(fr, []).append(b)
    return by_frame

def load_final_result(path: str) -> dict[int, list[dict]]:
    """
    Mode H: load final_result.json → {frame_no: [obj, …]}
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    by_frame: dict[int, list[dict]] = {}
    for entry in data.get("tracks", []):
        fr = entry.get("fr")
        objs = entry.get("obj", [])
        by_frame[fr] = objs
    return by_frame


# ───────────────────── colours & draw helpers ─────────────────────

COLMAP = {
    "black":    (0, 0, 0),
    "red":      (0, 0, 255),
    "purple":   (128, 0, 128),
    "darkblue": (128, 0, 0),
    "orange":   (0, 165, 255),
    "yellow":   (0, 255, 255),
    "green":    (0, 255, 0),
    "cyan":     (255, 255, 0),
    "magenta":  (255, 0, 255),
    "blue":     (255, 0, 0),
}
PINK  = (255, 0, 255)
WHITE = (255, 255, 255)

# Team colors: 1→green, 2→white, 3→yellow
TEAM_COLORS = {
    1: COLMAP["green"],
    2: WHITE,
    3: COLMAP["yellow"],
}

def colour_for_cluster(cid: int):
    palette = [COLMAP["black"], COLMAP["red"], COLMAP["purple"],
               COLMAP["darkblue"], COLMAP["orange"]]
    return palette[cid % len(palette)]

def draw_boxes(img, objs, colour):
    for o in objs:
        x1,y1,x2,y2 = map(int, o.get("bbox",[0,0,0,0]))
        cv2.rectangle(img, (x1,y1),(x2,y2), colour, 2)
        label = None
        if o.get("cluster") is not None:
            label = str(o["cluster"])
        elif o.get("id") is not None:
            label = str(o["id"])
        if label:
            cv2.putText(img, label, (x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
        tc = o.get("t_c")
        if tc is not None:
            cv2.putText(img, f"{tc[0]:.1f}", (x2+5,y1+15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, PINK, 1)

def draw_dots(img, dots, radius=2):
    for d in dots:
        x,y = int(d.get("x",0)), int(d.get("y",0))
        col = COLMAP.get(d.get("color","").lower(), COLMAP["green"])
        cv2.circle(img, (x,y), radius, col, -1)
        if d.get("id") is not None:
            cv2.putText(img, str(d["id"]),
                        (x+radius+2, y-radius-2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)

def draw_clustered_dots(img, clusters, radius=2):
    for cid, dots in clusters.items():
        col = colour_for_cluster(cid)
        first = True
        for d in dots:
            x,y = int(d.get("x",0)), int(d.get("y",0))
            cv2.circle(img, (x,y), radius, col, -1)
            if first:
                cv2.putText(img, f"C{cid}", (x+radius+2, y-radius-2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
                first = False

def draw_selected_boxes(img, sel, radius=2):
    for b in sel:
        x1,y1,x2,y2 = map(int, b.get("bbox",[0,0,0,0]))
        col = COLMAP.get(b.get("color","").lower(), COLMAP["cyan"])
        cv2.rectangle(img,(x1,y1),(x2,y2),col,2)
        cv2.putText(img, str(b.get("cluster","")),
                    (x1,y1-5), cv2.FONT_HERSHEY_SIMPLEX,0.5,col,1)

def draw_team_boxes(img, team_boxes, radius=2):
    """
    Mode G & H: draw per-team colored boxes:
    Team 1 → green, Team 2 → white, Team 3 → yellow.
    ID label in same color.
    """
    for b in team_boxes:
        x1,y1,x2,y2 = map(int, b.get("bbox",[0,0,0,0]))
        team = b.get("team", 0)
        col = TEAM_COLORS.get(team, WHITE)
        cv2.rectangle(img, (x1,y1), (x2,y2), col, 2)
        lbl = b.get("id")
        if lbl is not None:
            cv2.putText(img, str(lbl),
                        (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, col, 1)

def overlay_text(img, idx, total, mode):
    cv2.putText(img, f"{idx:06d}/{total-1:06d}",
                (10,25), cv2.FONT_HERSHEY_SIMPLEX,0.8,WHITE,2)
    cv2.putText(img, f"MODE {mode}",
                (10,55), cv2.FONT_HERSHEY_SIMPLEX,0.8,COLMAP["yellow"],2)


# ───────────────────── main ─────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",   required=True)
    ap.add_argument("--json",    required=True)
    ap.add_argument("--json2")
    ap.add_argument("--dots")
    ap.add_argument("--clusters")
    ap.add_argument("--selected")
    ap.add_argument("--teams")
    ap.add_argument("--final",   help="final_result.json for Mode H")
    args = ap.parse_args()

    video = Path(args.video)
    if not video.is_file() or not Path(args.json).is_file():
        sys.exit("❌ video or --json not found")

    A = load_bytetrack(args.json)
    B = load_boxes(args.json2)             if args.json2    else None
    C = load_dots(args.dots)               if args.dots     else None
    D = load_clustered_dots(args.clusters) if args.clusters else None
    E = load_selected_boxes(args.selected) if args.selected else None

    F_flat = []
    if E:
        for lst in E.values():
            F_flat.extend(lst)

    G_map = load_team_boxes_by_frame(args.teams) if args.teams else {}
    H_map = load_final_result(args.final)       if args.final else {}

    modes = ["A"]
    if B:      modes.append("B")
    if C:      modes.append("C")
    if D:      modes.append("D")
    if E:      modes.append("E")
    if F_flat: modes.append("F")
    if G_map:  modes.append("G")
    if H_map:  modes.append("H")
    mi = 0

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        sys.exit("❌ cannot open video")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    win = "viewer"
    cv2.namedWindow(win)
    last = None

    def refresh(pos):
        nonlocal last
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, frame = cap.read()
        if not ok:
            return

        mode = modes[mi]
        if mode == "A":
            draw_boxes(frame, A.get(pos, []), COLMAP["green"])
        elif mode == "B":
            draw_boxes(frame, B.get(pos, []) if B else [], COLMAP["yellow"])
        elif mode == "C":
            if C: draw_dots(frame, C)
        elif mode == "D":
            if D: draw_clustered_dots(frame, D)
        elif mode == "E":
            sel = E.get(pos, []) if E else []
            draw_selected_boxes(frame, sel)
        elif mode == "F":
            draw_selected_boxes(frame, F_flat)
        elif mode == "G":
            boxes = G_map.get(pos)
            if boxes:
                draw_team_boxes(frame, boxes)
            else:
                cv2.putText(
                    frame,
                    "No team-colored boxes on this frame",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )
        else:  # mode == "H"
            boxes = H_map.get(pos, [])
            if boxes:
                draw_team_boxes(frame, boxes)
            else:
                cv2.putText(
                    frame,
                    "No final mappings on this frame",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

        overlay_text(frame, pos, total, mode)
        cv2.imshow(win, frame)
        last = frame

    cv2.createTrackbar("frame", win, 0, total-1, refresh)
    refresh(0)

    playing = False
    while True:
        k = cv2.waitKey(20) & 0xFF
        if   k == ord('q'):
            break
        elif k == ord(' '):
            playing = not playing
        elif k == ord('s') and last is not None:
            idx = cv2.getTrackbarPos("frame", win)
            cv2.imwrite(f"frame_{idx:06d}_m{modes[mi]}.png", last)
        elif k in (81,82,ord('a')):
            p = max(cv2.getTrackbarPos("frame", win) - 1, 0)
            cv2.setTrackbarPos("frame", win, p)
        elif k in (83,84,ord('d')):
            p = min(cv2.getTrackbarPos("frame", win) + 1, total - 1)
            cv2.setTrackbarPos("frame", win, p)
        elif k == ord('z'):
            p = cv2.getTrackbarPos("frame", win)
            cv2.setTrackbarPos("frame", win, max(((p - 1)//100)*100, 0))
        elif k == ord('c'):
            p = cv2.getTrackbarPos("frame", win)
            cv2.setTrackbarPos("frame", win, min(((p//100)+1)*100, total-1))
        elif k == ord('g'):
            try:
                v = int(input(f"frame 0-{total-1}: "))
                if 0 <= v < total:
                    cv2.setTrackbarPos("frame", win, v)
            except ValueError:
                pass
        elif k == ord('m') and len(modes) > 1:
            mi = (mi + 1) % len(modes)
            refresh(cv2.getTrackbarPos("frame", win))

        if playing:
            p = cv2.getTrackbarPos("frame", win)
            if p + 1 < total:
                cv2.setTrackbarPos("frame", win, p + 1)
            else:
                playing = False

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
