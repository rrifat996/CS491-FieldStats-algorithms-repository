#!/usr/bin/env python3
"""
visualize_matching_logic.py · Demonstrate and visualize the formation matching pipeline

This standalone script walks through:
  1. Generating slot templates for a given formation
  2. Sampling synthetic data around each slot to learn means and covariances
  3. Computing the Mahalanobis-distance cost matrix for a set of detections
  4. Running the Hungarian assignment
  5. Plotting each step for clarity

Usage:
  python visualize_matching_logic.py
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

# — PARAMETERS —
WIDTH, HEIGHT = 740, 300
N_PLAYERS = 11
N_TRAIN = 50
EPS = 1e-6

# Example formation: 1-4-2-3-1 (GK, DEF4, DM2, AM3, ST1)
FORMATION = [1, 4, 1, 3, 2]

# Generate slot names for clarity
def generate_slot_names(formation):
    names = ["GK"]
    # Defense
    defs = ["LB","LCB","RCB","RB"]
    names += defs[:formation[1]]
    # DM
    dm = formation[2]
    if dm == 1:
        names.append("CDM")
    else:
        names += [f"DM{i+1}" for i in range(dm)]
    # AM
    am = formation[3]
    if am == 1:
        names.append("CAM")
    else:
        names += [f"AM{i+1}" for i in range(am)]
    # ST
    st = formation[4]
    if st == 1:
        names.append("ST")
    else:
        names += [f"ST{i+1}" for i in range(st)]
    return names

slot_names = generate_slot_names(FORMATION)

# Build ideal slot coordinates
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


template = build_template_slots(FORMATION)

# Plot the template
plt.figure(figsize=(6,3))
plt.scatter(template[:,0], template[:,1], c='red', marker='x')
for i, name in enumerate(slot_names):
    plt.text(template[i,0]+5, template[i,1]+5, name)
plt.title('Ideal Slot Template')
plt.xlim(0, WIDTH); plt.ylim(0, HEIGHT)
plt.gca().set_aspect('equal')
plt.show()

# Generate synthetic samples around each slot
def generate_samples(template):
    samples = []
    for i, center in enumerate(template):
        cov = np.diag([((0.06*WIDTH + (i/N_PLAYERS)*0.1*WIDTH)**2),
                       ((0.06*HEIGHT + (i/N_PLAYERS)*0.06*HEIGHT)**2)])
        pts = np.random.multivariate_normal(center, cov, size=N_TRAIN)
        samples.append(pts)
    return np.array(samples)

samples = generate_samples(template)

# Compute and plot covariance ellipses
plt.figure(figsize=(6,3))
plt.title('Samples around each slot')
for i in range(N_PLAYERS):
    pts = samples[i]
    plt.scatter(pts[:,0], pts[:,1], s=5, alpha=0.3)
plt.scatter(template[:,0], template[:,1], c='black', marker='x')
plt.xlim(0, WIDTH); plt.ylim(0, HEIGHT)
plt.gca().set_aspect('equal')
plt.show()

# Compute means and inverted covariances
mus = samples.mean(axis=1)
Sigmas = np.array([np.cov(samples[i].T, bias=False) for i in range(N_PLAYERS)])
Sigma_invs = np.array([np.linalg.inv(Sigmas[i] + EPS*np.eye(2)) for i in range(N_PLAYERS)])

# Create a noisy detection set by jittering the means
detections = mus + np.random.normal(scale=10, size=mus.shape)

# Plot detections
plt.figure(figsize=(6,3))
plt.scatter(detections[:,0], detections[:,1], c='blue', marker='o')
plt.title('Noisy Detections')
plt.xlim(0, WIDTH); plt.ylim(0, HEIGHT)
plt.gca().set_aspect('equal')
plt.show()

# Build cost matrix (Mahalanobis distances)
cost = np.zeros((N_PLAYERS, N_PLAYERS))
for i in range(N_PLAYERS):
    diff = detections - mus[i]
    cost[i] = np.sum((diff @ Sigma_invs[i]) * diff, axis=1)

# Display cost matrix as heatmap
plt.figure(figsize=(5,4))
plt.imshow(cost, cmap='viridis')
plt.colorbar(label='Cost')
plt.xlabel('Detection Index')
plt.ylabel('Slot Index')
plt.title('Cost Matrix')
plt.show()

# Run Hungarian matching
rows, cols = linear_sum_assignment(cost)
print('Assigned pairs (slot → detection):', list(zip(rows, cols)))

# Plot final assignment
plt.figure(figsize=(6,3))
plt.scatter(detections[:,0], detections[:,1], c='blue', label='detections')
plt.scatter(mus[:,0], mus[:,1], c='orange', marker='x', label='slot means')
for s, d in zip(rows, cols):
    plt.plot([mus[s,0], detections[d,0]], [mus[s,1], detections[d,1]], 'k-')
    plt.text(detections[d,0]+5, detections[d,1]+5, slot_names[s])
plt.title('Final Hungarian Assignment')
plt.xlim(0, WIDTH); plt.ylim(0, HEIGHT)
plt.gca().set_aspect('equal')
plt.legend()
plt.show()