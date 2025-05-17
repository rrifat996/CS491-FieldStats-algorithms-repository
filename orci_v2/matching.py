#!/usr/bin/env python3
"""
matching.py   ·  Multi-frame formation matcher

Reads an input JSON (single entry or list of entries), where each entry is:
  {
    "reference_frame": <int>,
    "bboxes": [ { "t_c": [x, y], "team": 1|2|3, ... }, … ]
  }

For each entry:
  • Runs Hungarian matching for Team 1 and Team 2 according to custom formations
    – Team 1 formation: 1-4-2-3-1 (GK, 4 DEF, 2 DM, 3 AM, 1 ST), mirrored
    – Team 2 formation: 1-4-1-2-3 (GK, 4 DEF, 1 DM, 2 AM, 3 ST)
    – Shows the plot only for the first entry
    – Prints “Completed match at frame X” thereafter
  • Annotates each bbox in-place:
       original_id → the old id (cluster or earlier id)
       id           → slot index for T1/T2, or 999 for team 3
       slot         → slot name (e.g. "GK", "LB", …)

Writes out a list of all enriched entries and also saves ideal slot positions per team.
"""
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from pathlib import Path

# — PARAMETERS & FORMATIONS —
WIDTH, HEIGHT = 740, 300
N_PLAYERS = 11
N_TRAIN   = 50

# Formation vectors: [GK, DEF, DM, AM, ST]
FORMATION_TEAM1 = [1, 4, 2, 3, 1]
FORMATION_TEAM2 = [1, 4, 1, 2, 3]


def generate_slot_names(formation):
    names = []
    # GK
    names.append("GK")
    # DEF
    defs = ["LB", "LCB", "RCB", "RB"]
    names.extend(defs[:formation[1]])
    # DM
    dm = formation[2]
    if dm == 1:
        names.append("CDM")
    elif dm == 2:
        names.extend(["LDM", "RDM"])
    elif dm == 3:
        names.extend(["LDM", "CDM", "RDM"])
    else:
        names.extend([f"DM{i}" for i in range(1, dm+1)])
    # AM
    am = formation[3]
    if am == 1:
        names.append("CAM")
    elif am == 2:
        names.extend(["LAM", "RAM"])
    elif am == 3:
        names.extend(["LAM", "CAM", "RAM"])
    else:
        names.extend([f"AM{i}" for i in range(1, am+1)])
    # ST
    st = formation[4]
    if st == 1:
        names.append("ST")
    elif st == 2:
        names.extend(["LS", "RS"])
    elif st == 3:
        names.extend(["LS", "ST", "RS"])
    else:
        names.extend([f"ST{i}" for i in range(1, st+1)])
    return names

slot_names1 = generate_slot_names(FORMATION_TEAM1)
slot_names2 = generate_slot_names(FORMATION_TEAM2)


def build_template_slots(formation):
    x_fracs = [0.05, 0.20, 0.40, 0.70, 0.90]
    xs, ys = [], []

    for i, grp in enumerate(formation):
        x = WIDTH * x_fracs[i]

        if i == 0:
            # goalkeeper sits in the very middle
            y_positions = [HEIGHT / 2]
        else:
            if grp == 1:
                # single slot → dead center
                y_positions = [HEIGHT / 2]
            else:
                # evenly space grp slots *around* center,
                # with no slot touching the very edge:
                step = HEIGHT / (grp + 1)
                # positions: step, 2*step, …, grp*step
                y_positions = list(step * np.arange(1, grp+1))

        xs += [x] * grp
        ys += y_positions

    return np.column_stack([xs, ys])

# Build templates and statistical models for each team
template1 = build_template_slots(FORMATION_TEAM1)
template2 = build_template_slots(FORMATION_TEAM2)


def generate_model(template):
    training = []
    for i in range(N_PLAYERS):
        cov_x = (0.04*WIDTH + (i/N_PLAYERS)*0.06*WIDTH)**2
        cov_y = (0.04*HEIGHT + (i/N_PLAYERS)*0.06*HEIGHT)**2
        Sigma_true = np.diag([cov_x, cov_y])
        smp = np.random.multivariate_normal(template[i], Sigma_true, size=N_TRAIN)
        training.append(smp)
    training = np.array(training)
    mu = training.mean(axis=1)
    Sigma = np.array([np.cov(training[i].T, bias=False) for i in range(N_PLAYERS)])
    return mu, Sigma

mu1, Sigma1 = generate_model(template1)
mu2, Sigma2 = generate_model(template2)
eps = 1e-6
Sigma_inv1 = np.array([np.linalg.inv(Sigma1[i] + eps*np.eye(2)) for i in range(N_PLAYERS)])
Sigma_inv2 = np.array([np.linalg.inv(Sigma2[i] + eps*np.eye(2)) for i in range(N_PLAYERS)])
# Mirror team1 horizontally
mu1_flipped = mu1.copy()
mu1_flipped[:, 0] = WIDTH - mu1_flipped[:, 0]


def build_cost_matrix(points, team):
    if team == 1:
        mu_loc, Sinv = mu1_flipped, Sigma_inv1
    else:
        mu_loc, Sinv = mu2, Sigma_inv2
    cost = np.zeros((N_PLAYERS, N_PLAYERS))
    for i in range(N_PLAYERS):
        d = points - mu_loc[i]
        cost[i] = np.sum((d @ Sinv[i]) * d, axis=1)
    return cost, mu_loc


def match_and_plot(points, title, team):
    cost, mu_loc = build_cost_matrix(points, team)
    rows, cols = linear_sum_assignment(cost)
    plt.figure()
    plt.scatter(points[:,0], points[:,1], marker='o', label='players')
    plt.scatter(mu_loc[:,0], mu_loc[:,1], marker='x', label='slots')
    names = slot_names1 if team==1 else slot_names2
    for s, p in zip(rows, cols):
        plt.plot([mu_loc[s,0], points[p,0]], [mu_loc[s,1], points[p,1]], 'k-')
        plt.text(points[p,0]+5, points[p,1]+5, names[s], fontsize=9)
    plt.title(title)
    plt.xlim(0, WIDTH); plt.ylim(0, HEIGHT)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()


def load_teams_tcs_entry(entry):
    by_team = {1: [], 2: []}
    for obj in entry.get("bboxes", []):
        t = obj.get("team"); tc = obj.get("t_c")
        if t in (1,2) and isinstance(tc, list) and len(tc)==2:
            by_team[t].append((tc, obj))
    for t in (1,2):
        if len(by_team[t]) != N_PLAYERS:
            rf = entry.get("reference_frame","?")
            raise ValueError(f"Expected {N_PLAYERS} players for team {t} at frame {rf}, got {len(by_team[t])}")
    pts1, objs1 = zip(*by_team[1]); pts2, objs2 = zip(*by_team[2])
    return np.array(pts1), np.array(pts2), entry, list(objs1), list(objs2)


def process_entry(entry, show_plots):
    t1, t2, full, objs1, objs2 = load_teams_tcs_entry(entry)
    if show_plots:
        match_and_plot(t1, "Team 1 (1-4-2-3-1, flipped)", team=1)
        match_and_plot(t2, "Team 2 (1-4-1-2-3)", team=2)
        plt.show()
    else:
        print(f"Completed match at frame {entry.get('reference_frame','?')}")

    def assign_slots(pts, objs, team):
        cost, mu_loc = build_cost_matrix(pts, team)
        rows, cols = linear_sum_assignment(cost)
        names = slot_names1 if team==1 else slot_names2
        for s, p in zip(rows, cols):
            obj = objs[p]
            obj["original_id"] = obj.get("id", obj.get("cluster"))
            obj["id"] = s
            obj["slot"] = names[s]

    assign_slots(t1, objs1, team=1)
    assign_slots(t2, objs2, team=2)

    # team 3 unchanged
    for obj in full.get("bboxes", []):
        if obj.get("team") == 3:
            obj["original_id"] = obj.get("id", obj.get("cluster"))
            obj["id"] = 999

    return full


def main():
    p = argparse.ArgumentParser()
    p.add_argument("json_in", type=Path, nargs="?", default=Path("output.json"))
    p.add_argument("--json_out", type=Path, default=Path("matched_output.json"))
    args = p.parse_args()

    raw = json.load(args.json_in.open())
    entries = raw if isinstance(raw, list) else [raw]
    out = []
    for i, ent in enumerate(entries):
        out.append(process_entry(ent, show_plots=(i==0)))

    # Save matched entries
    with args.json_out.open("w") as fp:
        json.dump(out, fp, indent=2, default=int)
    print(f"✓ Saved {len(out)} frame(s) → {args.json_out}")

    # Also save ideal slot positions for each team, including slot id
    ideal_positions = {
        "team1": [
            {"id": i, "slot": slot_names1[i], "position": [float(mu1_flipped[i,0]), float(mu1_flipped[i,1])]}
            for i in range(N_PLAYERS)
        ],
        "team2": [
            {"id": i, "slot": slot_names2[i], "position": [float(mu2[i,0]), float(mu2[i,1])]}
            for i in range(N_PLAYERS)
        ],
    }
    ideal_path = args.json_out.parent / "ideal_positions.json"
    with ideal_path.open("w") as fp2:
        json.dump(ideal_positions, fp2, indent=2)
    print(f"✓ Saved ideal positions → {ideal_path}")

if __name__ == "__main__":
    main()
