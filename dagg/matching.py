import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment

# PARAMETERS
WIDTH, HEIGHT = 400, 300
N_PLAYERS = 11
N_TRAIN = 50  # synthetic training samples per slot

# DEFINE FORMATION SLOTS (4-3-2-1 + GK)
slot_names = [
    "GK", "LB", "LCB", "RCB", "RB",
    "LM", "CM", "RM",
    "LAM", "RAM",
    "ST"
]
slot_x = (
    [0.05*WIDTH] +      # GK
    [0.20*WIDTH]*4 +    # Defenders
    [0.40*WIDTH]*3 +    # Midfielders
    [0.70*WIDTH]*2 +    # Attacking mids
    [0.90*WIDTH]        # Striker
)
y_gk      = [0.50*HEIGHT]
y_defs    = np.linspace(0.10*HEIGHT, 0.90*HEIGHT, 4)
y_mids    = np.linspace(0.20*HEIGHT, 0.80*HEIGHT, 3)
y_atts    = np.linspace(0.30*HEIGHT, 0.70*HEIGHT, 2)
y_striker = [0.50*HEIGHT]
slot_y = np.concatenate([y_gk, y_defs, y_mids, y_atts, y_striker])

template_slots = np.column_stack([slot_x, slot_y])

# SYNTHETIC TRAINING: generate samples around each slot with anisotropic covariances
training = []
for i in range(N_PLAYERS):
    # set varying covariances for demo
    cov_x = (0.04*WIDTH + (i/N_PLAYERS)*0.06*WIDTH)**2
    cov_y = (0.04*HEIGHT + (i/N_PLAYERS)*0.06*HEIGHT)**2
    Sigma_true = np.diag([cov_x, cov_y])
    samples = np.random.multivariate_normal(template_slots[i], Sigma_true, size=N_TRAIN)
    training.append(samples)
training = np.array(training)  # shape (11, N_TRAIN, 2)

# COMPUTE MU & SIGMA from training
mu = training.mean(axis=1)  # learned slot centers
Sigma = np.array([np.cov(training[i].T, bias=False) for i in range(N_PLAYERS)])
# regularize + invert
epsilon = 1e-6
Sigma_inv = np.array([np.linalg.inv(Sigma[i] + epsilon*np.eye(2)) for i in range(N_PLAYERS)])

# GENERATE TEST POINTS
points = np.column_stack([np.random.rand(N_PLAYERS)*WIDTH,
                          np.random.rand(N_PLAYERS)*HEIGHT])

# BUILD MAHALANOBIS COST MATRIX (squared distance)
cost_matrix = np.zeros((N_PLAYERS, N_PLAYERS))
for i in range(N_PLAYERS):
    diffs = points - mu[i]  # shape (11,2)
    cost_matrix[i] = np.sum((diffs @ Sigma_inv[i]) * diffs, axis=1)

# ASSIGN via Hungarian algorithm
row_ind, col_ind = linear_sum_assignment(cost_matrix)

# PLOT RESULTS
plt.figure(figsize=(8, 6))
# test points
plt.scatter(points[:,0], points[:,1], marker='o')
# learned slot centers
plt.scatter(mu[:,0], mu[:,1], marker='x')

# connect assignments and annotate
for slot_idx, point_idx in zip(row_ind, col_ind):
    plt.plot([mu[slot_idx,0], points[point_idx,0]],
             [mu[slot_idx,1], points[point_idx,1]])
    plt.text(points[point_idx,0] + 5, points[point_idx,1] + 5,
             slot_names[slot_idx], fontsize=9)

plt.xlim(0, WIDTH)
plt.ylim(0, HEIGHT)
plt.xlabel("Pitch Width")
plt.ylabel("Pitch Height")
plt.title("Formation Matching with Mahalanobis Cost")
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
