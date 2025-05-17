import json
import numpy as np
from scipy.optimize import linear_sum_assignment
import formations 
def get_player_positions_next(data, split_frame):
    """
    Returns a dictionary mapping each player id (int) to its position (x, y) at the split frame.
    
    For each player (skipping object id 23), this function checks the player's paths:
      - If the split frame falls within a path (i.e. between the path's start and end frame),
        the corresponding position in that path is returned.
      - Otherwise, if no path contains the split frame, it looks for the next path (one with
        a start frame greater than the split frame) and returns its first position.
    
    Parameters:
        data (dict): The data containing player objects and their paths.
        split_frame (int): The frame at which to retrieve the position.
        
    Returns:
        dict: A dictionary mapping player ids (int) to their [x, y] positions.
    """
    positions = {}
    for obj_id, obj in data.items():

        chosen_position = None
        candidate_path = None
        candidate_start_frame = None

        for path in obj['paths']:
            # Determine the start and end frame for this path.
            start_frame = path['last_seen_frame'] - (len(path['path']) - 1)
            end_frame = path['last_seen_frame']
            
            # If the split frame falls within this path, choose that position.
            if start_frame <= split_frame <= end_frame:
                frame_offset = split_frame - start_frame
                chosen_position = path['path'][frame_offset][1:]
                break  # Use the first matching path.
            
            # Otherwise, if the split frame is before the path starts, consider it as a candidate.
            if split_frame < start_frame:
                # Choose the candidate with the smallest start_frame (i.e. earliest upcoming path).
                if candidate_start_frame is None or start_frame < candidate_start_frame:
                    candidate_start_frame = start_frame
                    candidate_path = path

        # If no path contained the split frame, use the next path's first position (if available).
        if chosen_position is None and candidate_path is not None:
            chosen_position = candidate_path['path'][0][1:]
            
        if chosen_position:
            positions[int(obj_id)] = chosen_position
            
    return positions


    return {"left": leftmost_id, "right": rightmost_id}
def get_next_extreme_positions(data, split_frame):
    """
    Uses get_player_positions_next to determine which object is the leftmost (smallest x)
    and which is the rightmost (largest x) at the split frame, based on the next available positions.
    
    Parameters:
        data (dict): The data containing player objects and their paths.
        split_frame (int): The frame at which to retrieve the next positions.
        
    Returns:
        dict: A dictionary with keys "left" and "right" mapping to the respective object ids.
    """
    positions = get_player_positions_next(data, split_frame)
    
    leftmost_id = None
    rightmost_id = None
    leftmost_x = float('inf')
    rightmost_x = float('-inf')
    
    for obj_id, pos in positions.items():
        x, _ = pos
        if x < leftmost_x:
            leftmost_x = x
            leftmost_id = obj_id
        if x > rightmost_x:
            rightmost_x = x
            rightmost_id = obj_id

    return {"left": leftmost_id, "right": rightmost_id}

def get_player_team(data, split_frame):
    """
    Returns a dictionary mapping each player id (int) to its team (from the selected path)
    at the split frame, using the next available path's team if the split frame is not within any path.
    
    Parameters:
        data (dict): The data containing player objects and their paths.
        split_frame (int): The frame at which to determine the team.
        
    Returns:
        dict: A dictionary mapping player ids (int) to their team index.
    """
    teams = {}
    for obj_id, obj in data.items():
       
        team_val = None
        candidate_path = None
        candidate_start = None
        
        for path in obj['paths']:
            start_frame = path['last_seen_frame'] - (len(path['path']) - 1)
            end_frame = path['last_seen_frame']
            
            # Check if split frame is within this path's interval
            if start_frame <= split_frame <= end_frame:
                team_val = path.get('team_index')
                break  # Use the first matching path
            
            # Track the earliest next path (start_frame > split_frame)
            if start_frame > split_frame:
                if candidate_start is None or start_frame < candidate_start:
                    candidate_start = start_frame
                    candidate_path = path
        
        # If no overlapping path, use the candidate next path
        if team_val is None and candidate_path is not None:
            team_val = candidate_path.get('team_index')
        
        if team_val is not None:
            teams[int(obj_id)] = team_val
    
    return teams


def classify_non_extreme_players(data, split_frame):
    """
    For the 21 non-extreme players (i.e. excluding the leftmost and rightmost players),
    classify them into three teams based on the team value from their split-frame path.
    
    The expected grouping is:
      - Team -1: 10 players
      - Team  0:  1 player
      - Team  1: 10 players
    
    If the initial grouping does not match the expected counts (i.e. one group is off by one),
    one extra (overflowing) element from the surplus group is swapped into the deficit group.
    
    Returns a dict with keys -1, 0, and 1 mapping to lists of object ids.
    """
    # Get the extreme players (these will be excluded)
    extremes = get_next_extreme_positions(data, split_frame)
    left_extreme = extremes.get("left")
    right_extreme = extremes.get("right")
    
    # Get team values for all players at the split frame.
    teams_all = get_player_team(data, split_frame)
    # Exclude the extreme players
    non_extreme_teams = {pid: team for pid, team in teams_all.items() if pid not in [left_extreme, right_extreme]}
    

    print(teams_all)
    print(non_extreme_teams)   

    if len(non_extreme_teams) != 21:
        raise ValueError(f"Expected 21 non-extreme players, got {len(non_extreme_teams)}")
    
    # Group players by their team value.
    team_groups = { -1: [], 0: [], 1: [] }
    for pid, team_val in non_extreme_teams.items():
        if team_val in team_groups:
            team_groups[team_val].append(pid)
        else:
            # In case an unexpected team value is encountered, you might decide to handle it.
            pass

    # Expected counts:
    expected_counts = { -1: 10, 0: 1, 1: 10 }

    if len(team_groups[0]) == 0:
        max_team = max(team_groups, key=lambda t: len(team_groups[t]) if t != 0 else -1)
        if team_groups[max_team]:  # Ensure the team has at least one player to move
            player_to_move = team_groups[max_team].pop()
            team_groups[0].append(player_to_move)

    diff = { team: len(team_groups[team]) - expected_counts[team] for team in team_groups }

    # Continue swapping until all differences are balanced (i.e. diff is 0 for every team)
    while any(d > 0 for d in diff.values()) and any(d < 0 for d in diff.values()):
        # Identify teams with surplus and with deficit (allowing for differences larger than 1)
        surplus_teams = [team for team, d in diff.items() if d > 0]
        deficit_teams = [team for team, d in diff.items() if d < 0]
        
        # Choose one surplus and one deficit team (you could also iterate over all pairs if desired)
        surplus_team = surplus_teams[0]
        deficit_team = deficit_teams[0]
        
        # Move one element from the surplus team to the deficit team.
        if team_groups[surplus_team]:  # ensure there's at least one element to pop
            element_to_swap = team_groups[surplus_team].pop()
            team_groups[deficit_team].append(element_to_swap)
            
            # Update the differences.
            diff[surplus_team] -= 1
            diff[deficit_team] += 1
    
    print(expected_counts)
    # Final check that counts match the expected numbers.
    for team in expected_counts:
        if len(team_groups[team]) != expected_counts[team]:
            raise ValueError(
                f"After swapping, team {team} has {len(team_groups[team])} players; expected {expected_counts[team]}."
            )
    
    return team_groups

import numpy as np

def compute_id_cost_matrix(valid_ids):
    """
    Computes an n x n matrix of ID costs.
    
    For each pair (i, j) in valid_ids, the cost is defined as:
         id_cost = abs(int(valid_ids[i]) - int(valid_ids[j])) * id_weight
    
    Args:
        valid_ids (list): A list of player IDs (as strings or integers).
        id_weight (float): The weight applied to the absolute ID difference.
    
    Returns:
        np.ndarray: An n x n numpy array containing the ID cost for each pairing.
    """
    n = len(valid_ids)
    id_cost_matrix = np.zeros((n, n))
    for i, obj_id in enumerate(valid_ids):
        for j, target_id in enumerate(valid_ids):
            id_cost_matrix[i][j] = abs(int(obj_id) - int(target_id)) 
    return id_cost_matrix



def compute_distance_matrix(player_positions):
    """Compute the 11x11 Euclidean distance matrix for a given team's positions."""
    ids = sorted(player_positions.keys())  # Ensure ordered by ID
    n = len(ids)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:  # Avoid self-distance
                pos_i = player_positions[ids[i]]
                pos_j = player_positions[ids[j]]
                distance_matrix[i, j] = np.linalg.norm(np.array(pos_i) - np.array(pos_j))
    
    return distance_matrix

import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment

def final_index_matching(teams, player_positions,
                          next_left_extreme,
                          next_right_extreme,
                         fixed_self = 23):
    """
    Computes formation assignments (formation IDs) separately for the two teams.
    
    For team right (teams[1]) and team left (teams[-1]):
      - Extract each player's x coordinate.
      - Run k-means clustering with k=4.
      - Order the clusters from left to right based on cluster centers.
      - Build a cost matrix (squared difference between a player's x coordinate
        and the cluster center) and expand it so that each cluster gets the specified number of players.
      - Solve the assignment with the Hungarian algorithm.
      - Within each cluster, sort players by x coordinate and then assign formation IDs sequentially.
    
    Capacities:
      - For team right: capacities_right = [1, 2, 3, 4] (i.e. first cluster gets 1 slot, second gets 2, etc.)
      - For team left:  capacities_left  = [4, 3, 2, 1]
      
    Returns:
      A tuple (formation_ids_right, formation_ids_left) where each is a dictionary mapping
      player ID to its formation ID (ordered left-to-right).
    """
    import numpy as np
    from sklearn.cluster import KMeans
    from scipy.optimize import linear_sum_assignment

    # --- Get team positions ---
    # team_right: teams[1], team_left: teams[-1]
    team_right_pos = {player_id: player_positions[player_id]
                      for player_id in teams[1] if player_id in player_positions}
    team_left_pos = {player_id: player_positions[player_id]
                     for player_id in teams[-1] if player_id in player_positions}

    # Extract player IDs and their x coordinates (reshaped into a column vector).
    team_right_ids = list(team_right_pos.keys())
    team_left_ids = list(team_left_pos.keys())
    player_x_right = np.array([team_right_pos[pid][0] for pid in team_right_ids]).reshape(-1, 1)
    player_x_left = np.array([team_left_pos[pid][0] for pid in team_left_ids]).reshape(-1, 1)

    # --- k-means clustering on x coordinate ---
    # Run k-means with 4 clusters for each team.
    kmeans_right = KMeans(n_clusters=4, random_state=0).fit(player_x_right)
    kmeans_left = KMeans(n_clusters=4, random_state=0).fit(player_x_left)
    
    # Retrieve cluster centers.
    cluster_centers_right = kmeans_right.cluster_centers_.flatten()
    cluster_centers_left  = kmeans_left.cluster_centers_.flatten()

    # Order clusters by center (left-to-right) for each team.
    order_right = np.argsort(cluster_centers_right)
    sorted_centers_right = cluster_centers_right[order_right]
    
    order_left = np.argsort(cluster_centers_left)
    sorted_centers_left = cluster_centers_left[order_left]

    # Define capacities: different for each team.
    capacities_right = [1, 3, 2, 4]  # Right team: 1 slot in leftmost cluster, then 2, etc.
    capacities_left  = [4, 1, 2, 3]  # Left team: 4 slots in leftmost cluster, then 3, etc.

    def assign_formation_ids(player_ids, player_x, sorted_centers, capacities):
        """
        Given a list of player_ids and their x coordinates (as a 2D array),
        assign formation IDs using:
          - a cost matrix based on squared differences to sorted_centers,
          - expanding columns according to capacities,
          - Hungarian algorithm assignment,
          - and then sorting within each cluster by the x coordinate.
        
        Returns:
          formation_ids: a dict mapping player_id to its formation slot (an integer).
        """
        n_players = len(player_x)
        # Build cost matrix: cost[i, j] = squared difference between player's x and j-th ordered cluster center.
        cost_matrix = np.zeros((n_players, 4))
        for i in range(n_players):
            for j in range(4):
                cost_matrix[i, j] = (player_x[i][0] - sorted_centers[j]) ** 2

        # Expand cost matrix: duplicate each column j according to its capacity.
        expanded_cost = np.hstack([np.tile(cost_matrix[:, j:j+1], (1, cap))
                                   for j, cap in enumerate(capacities)])
        
        # Solve the assignment with the Hungarian algorithm.
        row_ind, col_ind = linear_sum_assignment(expanded_cost)
        
        # boundaries: cumulative indices where each cluster's columns end.
        boundaries = np.cumsum([0] + capacities)  # e.g., for capacities [1,2,3,4] -> [0, 1, 3, 6, 10]
        
        # Group assignments by cluster.
        assignments_by_cluster = {i: [] for i in range(4)}
        for r, c in zip(row_ind, col_ind):
            for cluster in range(4):
                if boundaries[cluster] <= c < boundaries[cluster + 1]:
                    assignments_by_cluster[cluster].append((player_ids[r], player_positions[player_ids[r]][1]))

                    break
        
        # Within each cluster, sort players by their y coordinate (increasing).
        for cluster in assignments_by_cluster:
            assignments_by_cluster[cluster] = sorted(assignments_by_cluster[cluster], key=lambda tup: tup[1])
        
        # Assign formation IDs in order: first cluster gets the first set of IDs, etc.
        formation_ids = {}
        current_id = 1
        for cluster in range(4):
            sorted_players = sorted(assignments_by_cluster[cluster], key=lambda tup: tup[1])   # Sort by increasing x

            for pid, _ in sorted_players:
                formation_ids[pid] = current_id
                current_id += 1
        return formation_ids

    # --- Compute formation assignments for both teams ---
    formation_ids_right = assign_formation_ids(team_right_ids, player_x_right, sorted_centers_right, capacities_right)
    formation_ids_left  = assign_formation_ids(team_left_ids,  player_x_left,  sorted_centers_left,  capacities_left)
    

    x = 1  # Example
    y = 11  # Example

    # Add x and y to the values of both maps
    updated_right = {k: v + x for k, v in formation_ids_right.items()}
    updated_left  = {k: v + y for k, v in formation_ids_left.items()}

    merged_keys = list(updated_right.keys()) + list(updated_left.keys())
    merged_values = list(updated_right.values()) + list(updated_left.values())

    print("Merged Keys:", merged_keys)
    print("Merged Values:", merged_values)

    final_match_before_all = merged_keys + [next_left_extreme, next_right_extreme, fixed_self]
    final_match_after_all = merged_values + [1, 22, 23]


    
    # Print out the assignments.
    print("Team Right Formation IDs (from left to right):")
    for pid in sorted(formation_ids_right, key=lambda x: formation_ids_right[x]):
        print(f"Player {pid}: Formation ID {formation_ids_right[pid]}")
    
    print("\nTeam Left Formation IDs (from left to right):")
    for pid in sorted(formation_ids_left, key=lambda x: formation_ids_left[x]):
        print(f"Player {pid}: Formation ID {formation_ids_left[pid]}")

    return final_match_before_all, final_match_after_all



def process_matching(input_file, output_file, t=2, fps=60):
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    split_interval = t * fps
    all_ids = sorted(data.keys(), key=lambda x: int(x))  # Maintain ID order
    
    max_frame = max(p['last_seen_frame'] for obj in data.values() for p in obj['paths'])
    split_frames = [i * split_interval for i in range(1, (max_frame // split_interval) + 1)]

    all_mappings = []

    for split_frame in split_frames:
        print(f"\nProcessing split frame {split_frame}")
        
        # Find closest paths around split frame for all IDs
        prev_paths = {}
        next_paths = {}
        
        for obj_id in all_ids:
            obj = data[obj_id]
            paths = obj['paths']
            
            # Find closest previous path (last_seen <= split_frame)
            prev_candidates = [p for p in paths if p['last_seen_frame'] <= split_frame]
            prev = max(prev_candidates, key=lambda x: x['last_seen_frame']) if prev_candidates else None
            
            # Find closest next path (starts after split_frame)
            next_candidates = []
            for p in paths:
                start_frame = p['last_seen_frame'] - (len(p['path']) - 1)
                if start_frame > split_frame:
                    next_candidates.append((start_frame, p))
            
            nxt = min(next_candidates, key=lambda x: x[0])[1] if next_candidates else None
            
            prev_paths[obj_id] = prev
            next_paths[obj_id] = nxt

        # Create cost matrix with weighted components
        valid_ids = [obj_id for obj_id in all_ids if prev_paths[obj_id] and next_paths[obj_id]]
        n = len(valid_ids)
        print("n :\n", n) 
        if n == 0:
            continue
        if n != 23:
            continue
        if split_frame > 9000:
            continue

        #cost_matrix += compute_id_cost_matrix(valid_ids, id_weight)
        player_positions_next = get_player_positions_next(data, split_frame)
        #print(len(player_positions))
        #print(len(player_positions_next))

        next_extremes = get_next_extreme_positions(data, split_frame)
        next_left_extreme = next_extremes.get("left")
        next_right_extreme = next_extremes.get("right")

        teams = classify_non_extreme_players(data, split_frame)
        fixed_self = teams[0][0]


        final_left, final_right = final_index_matching( teams, player_positions_next, next_left_extreme, next_right_extreme,  fixed_self )

        print("Team -1:", teams[-1])
        print("Team  0:", teams[0])
        print("Team  1:", teams[1])
        
        # Create ID mapping based on matches
        id_mapping = {}
        for i, j in zip(final_left, final_right):
            original_id = str(i)
            target_id   = str(j)
            id_mapping[original_id] = target_id

        # 2. Print the mappings to the console for clarity
        print(f"Mapped IDs at frame {split_frame}:")
        for orig, new in id_mapping.items():
            print(f"{orig} → {new}")

        # 3. Prepare the JSON structure
        mapping_data = {
            "frame": split_frame,
            "mapping": id_mapping
        }
        all_mappings.append(mapping_data)

    # 4. Save to JSON (no changes made to original data)
    with open(output_file, 'w') as f:
        json.dump(all_mappings, f, indent=4)
    print(f"\nMapping saved to {output_file}")

# Example usage with custom weights
process_matching('output.json', 'output2.json', t=5, fps=60)