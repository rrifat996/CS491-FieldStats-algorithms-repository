import cv2
import numpy as np
import json
from tqdm import tqdm
import random

# File paths
VIDEO_LEFT = "l5.mp4"
VIDEO_RIGHT = "r5.mp4"
OUTPUT_VIDEO = "output.mp4"
INPUT_JSON = "input2.json"
N_FRAMES = 100  # Number of frames to process

# Global variable to store the 4-gon points selected on the merged frame
selected_points = []

def transform_polygon_to_local(polygon, offset_x=0):
    """
    Shift the polygon's x coordinates by offset_x.
    This maps the polygon from the merged frame coordinates to the local (video) frame coordinates.
    """
    return [(pt[0] - offset_x, pt[1]) for pt in polygon]

def get_random_point_in_polygon(polygon, max_attempts=1000):
    """
    Returns a random point inside a polygon (list of (x, y) points).
    Uses the bounding rectangle of the polygon and then verifies if the point is inside.
    """
    poly_np = np.array(polygon, dtype=np.int32)
    min_x = np.min(poly_np[:, 0])
    max_x = np.max(poly_np[:, 0])
    min_y = np.min(poly_np[:, 1])
    max_y = np.max(poly_np[:, 1])
    
    for _ in range(max_attempts):
        rand_x = random.uniform(min_x, max_x)
        rand_y = random.uniform(min_y, max_y)
        if cv2.pointPolygonTest(poly_np, (rand_x, rand_y), False) >= 0:
            return np.array([rand_x, rand_y], dtype=np.float32)
    return np.array([(min_x + max_x) / 2, (min_y + max_y) / 2], dtype=np.float32)

def draw_bboxes(frame, objects, scale_x, scale_y, local_poly):
    # Color mapping
    color_map = {
        "blue": (255, 0, 0),
        "purple": (128, 0, 128),
        "red": (0, 0, 255),
        "orange": (0, 165, 255),
        "yellow": (0, 255, 255)
    }
    
    # First pass: draw bboxes that do not meet relocation condition
    for obj in objects:
        bbox = obj["bbox"]
        color_name = obj["color"]
        
        # Ensure t_c is a tuple of floats
        t_c_raw = obj.get("t_c", (0, 0))
        t_c = tuple(float(x) for x in t_c_raw)
        
        if color_name in color_map:
            color = color_map[color_name]
            
            # Scale bbox coordinates
            x1, y1, x2, y2 = bbox
            x1 = int(float(x1) * scale_x)
            y1 = int(float(y1) * scale_y)
            x2 = int(float(x2) * scale_x)
            y2 = int(float(y2) * scale_y)
            
            # Check if the bbox meets the general relocation condition
            condition_met = local_poly is not None and (0 < t_c[0] < 200 or t_c[0] > 300)
            
            # First pass: only draw bboxes that do NOT meet relocation condition
            if not condition_met:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    
    # Second pass: draw relocated bboxes that meet the condition
    for obj in objects:
        bbox = obj["bbox"]
        color_name = obj["color"]
        t_c_raw = obj.get("t_c", (0, 0))
        t_c = tuple(float(x) for x in t_c_raw)
        
        if color_name in color_map:
            color = color_map[color_name]
            
            # Scale bbox coordinates
            x1, y1, x2, y2 = bbox
            x1 = int(float(x1) * scale_x)
            y1 = int(float(y1) * scale_y)
            x2 = int(float(x2) * scale_x)
            y2 = int(float(y2) * scale_y)
            
            # Check if the bbox meets the general relocation condition
            condition_met = local_poly is not None and (0 < t_c[0] < 200 or t_c[0] > 300)
            
            # Second pass: only draw bboxes that meet relocation condition
            if condition_met:
                # Calculate bbox width and height to maintain
                bbox_width = x2 - x1
                bbox_height = y2 - y1
                
                # Get a random point in the local polygon
                random_center = get_random_point_in_polygon(local_poly)
                
                # Calculate new bbox coordinates centered at the random point
                new_x1 = int(random_center[0] - bbox_width / 2)
                new_y1 = int(random_center[1] - bbox_height / 2)
                new_x2 = new_x1 + bbox_width
                new_y2 = new_y1 + bbox_height
                
                # Draw the relocated bbox with original color
                cv2.rectangle(frame, (new_x1, new_y1), (new_x2, new_y2), color, 2)
                
                # Mark the random center point
                cv2.circle(frame, (int(random_center[0]), int(random_center[1])), 3, (0, 255, 0), -1)

                cx, cy = (new_x1 + new_x2) // 2, (new_y1 + new_y2) // 2
                cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1)  # Yellow dot
    
    return frame

def select_points(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(selected_points) < 4:
        selected_points.append((x, y))
        cv2.circle(param, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Select 4-gon", param)

def get_transform_selection(merged_frame):
    cv2.imshow("Select 4-gon", merged_frame)
    cv2.setMouseCallback("Select 4-gon", select_points, merged_frame)
    while len(selected_points) < 4:
        cv2.waitKey(1)
    cv2.destroyAllWindows()

def concatenate_videos(video1: str, video2: str, output: str, json_file: str, n_frames: int):
    cap1 = cv2.VideoCapture(video1)
    cap2 = cv2.VideoCapture(video2)
    
    if not cap1.isOpened() or not cap2.isOpened():
        print("Error: One or both video files could not be opened.")
        return

    # Original dimensions
    orig_width1 = int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height1 = int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
    orig_width2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Resize dimensions for each video (each is 500x750)
    new_width, new_height = 500, 750  
    scale_x1, scale_y1 = new_width / orig_width1, new_height / orig_height1
    scale_x2, scale_y2 = new_width / orig_width2, new_height / orig_height2
    
    with open(json_file, "r") as f:
        data = json.load(f)
    frame_data = {frame["fr"]: frame["obj"] for frame in data.get("frames", [])}
    
    fps = int(cap1.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output, fourcc, fps, (new_width*2, new_height))
    
    # Get the merged frame for polygon selection (merged frame is now 1000x750)
    middle_frame_idx = n_frames // 2
    cap1.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_idx)
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    
    if ret1 and ret2:
        frame1 = cv2.resize(frame1, (new_width, new_height))
        frame2 = cv2.resize(frame2, (new_width, new_height))
        merged_frame = np.hstack((frame1, frame2))
        get_transform_selection(merged_frame)
    else:
        print("Could not read middle frames for selection.")
    
    # Prepare local polygons for each video
    # For left video (blue/purple), the polygon is taken as is (from merged frame)
    local_poly_left = transform_polygon_to_local(selected_points, offset_x=0)
    # For right video (red/orange/yellow), subtract the left video width
    local_poly_right = transform_polygon_to_local(selected_points, offset_x=new_width)
    
    # Reset captures
    cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
    cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    for frame_idx in tqdm(range(n_frames), desc="Processing frames"):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        if not ret1 or not ret2:
            break
        
        frame1 = cv2.resize(frame1, (new_width, new_height))
        frame2 = cv2.resize(frame2, (new_width, new_height))
        
        if frame_idx in frame_data:
            objects = frame_data[frame_idx]
            # Process objects for left video (blue/purple)
            for obj in objects:
                if obj["color"] in ["blue", "purple"]:
                    frame1 = draw_bboxes(frame1, [obj], scale_x1, scale_y1, local_poly_left)
                elif obj["color"] in ["red", "orange", "yellow"]:
                    frame2 = draw_bboxes(frame2, [obj], scale_x2, scale_y2, local_poly_right)
        
        combined_frame = np.hstack((frame1, frame2))
        
        # Draw the (global) purple 4‑gon on the merged frame for reference
        pts = np.array(selected_points, dtype=np.int32)
        cv2.polylines(combined_frame, [pts], isClosed=True, color=(128, 0, 128), thickness=2)
        
        out.write(combined_frame)
    
    cap1.release()
    cap2.release()
    out.release()
    print(f"Video saved as {output}")

if __name__ == "__main__":
    concatenate_videos(VIDEO_LEFT, VIDEO_RIGHT, OUTPUT_VIDEO, INPUT_JSON, N_FRAMES)