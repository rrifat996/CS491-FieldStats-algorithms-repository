#!/usr/bin/env python3
import json
import random
from pathlib import Path


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def process_mapping(matched_output, result_json):
    # Build explicit per-frame maps from matched_output
    reference_explicit = {}
    reference_teams    = {}

    for entry in matched_output:
        frame = entry.get('reference_frame')
        if frame is None:
            continue
        reference_explicit.setdefault(frame, {})
        reference_teams.setdefault(frame, {})
        id_map   = reference_explicit[frame]
        team_map = reference_teams[frame]

        for bbox in entry.get('bboxes', []):
            orig = bbox.get('original_id')
            new  = bbox.get('id')
            team = bbox.get('team')
            if orig is None or new is None:
                continue
            # record only first-seen mapping
            if orig not in id_map:
                id_map[orig]   = new
                team_map[orig] = team

    # Sort reference frames for interval processing
    sorted_frames = sorted(reference_explicit.keys())

    # Apply mapping for each interval [current_ref, next_ref)
    for i, current_ref in enumerate(sorted_frames):
        next_ref = sorted_frames[i+1] if i+1 < len(sorted_frames) else None
        explicit_map = reference_explicit[current_ref]
        team_map     = reference_teams[current_ref]

        # Prepare pool of random IDs (0-22 minus explicitly used)
        used_ids      = set(explicit_map.values())
        available_ids = list(set(range(23)) - used_ids)
        random_map    = {}

        print(f"\nInterval {current_ref} to {next_ref}: explicit={explicit_map}")

        # Iterate through tracks in the interval
        for track in result_json.get('tracks', []):
            fr = track.get('fr')
            if fr is None or fr < current_ref or (next_ref is not None and fr >= next_ref):
                continue

            for obj in track.get('obj', []):
                orig_id = obj.get('id')

                if orig_id in explicit_map:
                    # use the explicit mapping
                    new_id = explicit_map[orig_id]
                    team   = team_map.get(orig_id)
                else:
                    # assign a random unused ID
                    if orig_id not in random_map:
                        choice = random.choice(available_ids)
                        available_ids.remove(choice)
                        random_map[orig_id] = choice
                    new_id = random_map[orig_id]
                    team   = None

                # overwrite with new ID
                obj['id'] = new_id

                # set or remove team field
                if team is not None:
                    obj['team'] = team
                else:
                    obj.pop('team', None)

        print(f"  random assignments: {random_map}")

    return result_json


def main():
    matched_output = load_json('matched_output.json')
    result_json    = load_json('result2.json')

    final_json = process_mapping(matched_output, result_json)
    save_json(final_json, 'final_result.json')
    print("\n✅ Processing complete. Saved to final_result.json")


if __name__ == "__main__":
    main()