#!/usr/bin/env python3
import json
import os
import sys

import numpy as np
from sklearn.cluster import KMeans

def main():
    input_path = "dots_120.json"
    output_path = "clusters.json"

    # 1. Load
    if not os.path.exists(input_path):
        print(f"Error: '{input_path}' not found.", file=sys.stderr)
        sys.exit(1)
    with open(input_path, "r") as f:
        data = json.load(f)

    # Validate
    if "dots" not in data or not isinstance(data["dots"], list):
        print("Error: input JSON must contain a 'dots' list.", file=sys.stderr)
        sys.exit(1)

    dots = data["dots"]

    # 2. Extract (x, y) for clustering
    coords = []
    for i, dot in enumerate(dots):
        try:
            x = float(dot["x"])
            y = float(dot["y"])
        except (KeyError, ValueError):
            print(f"Error: dot at index {i} missing valid 'x'/'y'.", file=sys.stderr)
            sys.exit(1)
        coords.append([x, y])
    coords = np.array(coords)

    # 3. Run KMeans
    k = 23
    kmeans = KMeans(n_clusters=k, random_state=0)
    labels = kmeans.fit_predict(coords)

    # 4. Attach cluster labels
    for dot, label in zip(dots, labels):
        dot["cluster"] = int(label)

    # 5. Write out
    out = {
        "reference_frame": data.get("reference_frame"),
        "dots": dots
    }
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Done: assigned {k} clusters and wrote to '{output_path}'.")

if __name__ == "__main__":
    main()
