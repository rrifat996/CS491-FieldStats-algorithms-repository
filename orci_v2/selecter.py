#!/usr/bin/env python3
"""
select_bboxes.py   (v3)

Given a JSON of clustered dots—either a single object or a list—with
fields:

    [ {"reference_frame": R, "dots": [ ... ]}, ... ]

this script selects one bbox per cluster (0–22) using colour priority:

    yellow > red > blue

and writes the results—preserving frame, colour, t_c, id, and bbox—for each
reference frame into an output JSON array.

Defaults:
    input:  clusters.json
    output: selected_bboxes.json

Usage:
    python select_bboxes.py
    python select_bboxes.py in.json out.json
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any

COLOUR_PRIORITY = ["yellow", "red", "blue"]
PRIORITY_MAP = {c: i for i, c in enumerate(COLOUR_PRIORITY)}


def load_data(path: Path) -> List[Dict[str, Any]]:
    """
    Load and normalize input to a list of entries with 'reference_frame' and 'dots'.
    """
    if not path.exists():
        sys.exit(f"Error: '{path}' not found.")
    with path.open("r", encoding="utf-8") as fp:
        data = json.load(fp)

    # Single object case
    if isinstance(data, dict) and "dots" in data and "reference_frame" in data:
        return [data]
    # Already a list
    if isinstance(data, list):
        return data

    sys.exit("Error: input must be a dict or list of dicts with 'reference_frame' and 'dots'.")


def choose_bbox_per_cluster(dots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    From a list of dots, pick one per cluster by colour priority.
    """
    clusters: Dict[int, List[Dict[str, Any]]] = {}
    for d in dots:
        cid = d.get("cluster")
        clusters.setdefault(cid, []).append(d)

    selected: List[Dict[str, Any]] = []
    for cid in range(23):  # clusters 0..22
        if cid not in clusters:
            sys.exit(f"No dots present for cluster {cid}")
        best = min(clusters[cid], key=lambda d: PRIORITY_MAP.get(d.get("color"), 99))
        selected.append({
            "cluster": cid,
            "frame":   best["frame"],
            "color":   best["color"],
            "bbox":    best["bbox"],
            "t_c":     best.get("t_c"),
            "id":      best.get("id"),
        })
    return selected


def main() -> None:
    in_path  = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("clusters.json")
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("selected_bboxes.json")

    entries = load_data(in_path)
    results: List[Dict[str, Any]] = []

    for entry in entries:
        dots = entry.get("dots", [])
        sel = choose_bbox_per_cluster(dots)
        results.append({
            "reference_frame": entry.get("reference_frame"),
            "bboxes": sel
        })

    with out_path.open("w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)

    print(f"[✓] Wrote selections for {len(results)} reference frames to '{out_path}'.")


if __name__ == "__main__":
    main()
