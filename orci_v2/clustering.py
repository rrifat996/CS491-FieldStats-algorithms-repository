#!/usr/bin/env python3
"""
cluster_dots.py

Given a JSON output from `extract_dots.py`—which may contain a list of
reference-frame entries—this script will run KMeans clustering on each
set of dots and produce a JSON file containing clusters for every
reference frame.

If input is a single object instead of a list, it will be handled uniformly.

Defaults:
    input:  dots_100.json
    output: clusters.json
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

import numpy as np
from sklearn.cluster import KMeans


def load_data(path: Path) -> List[Dict[str, Any]]:
    """
    Load input JSON and normalize to a list of {reference_frame:…, dots: […]}.
    """
    with path.open('r', encoding='utf-8') as f:
        data = json.load(f)

    # If single object, wrap in list
    if isinstance(data, dict) and 'dots' in data and 'reference_frame' in data:
        return [data]
    # If list of objects
    if isinstance(data, list):
        return data

    print("Error: input JSON must be a dict or list of dicts with 'reference_frame' and 'dots'", file=sys.stderr)
    sys.exit(1)


def cluster_for_entry(entry: Dict[str, Any], k: int = 23) -> Dict[str, Any]:
    """
    Perform KMeans clustering on one entry's dots and append 'cluster' labels.
    Returns a new dict with same structure.
    """
    dots = entry.get('dots', [])
    coords = []
    for i, dot in enumerate(dots):
        try:
            x = float(dot['x'])
            y = float(dot['y'])
        except (KeyError, ValueError):
            print(f"Error: dot at index {i} missing valid 'x'/'y'.", file=sys.stderr)
            sys.exit(1)
        coords.append([x, y])

    coords_arr = np.array(coords)
    if len(coords_arr) == 0:
        # No dots to cluster
        entry['clusters'] = []
        return entry

    kmeans = KMeans(n_clusters=min(k, len(coords_arr)), random_state=0)
    labels = kmeans.fit_predict(coords_arr)

    # Attach cluster labels
    for dot, label in zip(dots, labels):
        dot['cluster'] = int(label)

    return {
        'reference_frame': entry['reference_frame'],
        'dots': dots
    }


def main() -> None:
    input_path = Path('dots_100.json')
    output_path = Path('clusters.json')

    if not input_path.exists():
        print(f"Error: '{input_path}' not found.", file=sys.stderr)
        sys.exit(1)

    entries = load_data(input_path)
    results = []

    for entry in entries:
        clustered = cluster_for_entry(entry)
        results.append(clustered)

    # Write out
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)

    total_entries = len(results)
    total_clusters = sum(len(r['dots']) for r in results)
    print(f"Done: processed {total_entries} reference frames, wrote clusters to '{output_path}' ({total_clusters} points).")


if __name__ == '__main__':
    main()
