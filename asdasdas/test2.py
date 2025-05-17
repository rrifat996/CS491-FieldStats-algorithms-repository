import cv2
import numpy as np
from tqdm import tqdm
import json

INPUT_VIDEOS = ["right5.mp4", "left5shifted.mp4"]  # Input video fileapaths
OUTPUT_VIDEOS = ["r5.mp4", "l5.mp4"]          # Output video file paths
INPUT_JSON = "80_iou_compressed.json"  # Input JSON file path
OUTPUT_JSON = "input2.json"  # Output JSON file path
N_FRAMES = 5000  # Process first n frames
START_FRAME = 0  # Start from frame 100 (you can change this to any frame number)

# Display size for intermediate windows (height, width)
DISPLAY_SIZE = (600, 1200)

# Global variables for point selection
clicked_points = []
display_frame = None  # resized image for selection

# Transformation matrices for left and right videos
M_left = None
M_right = None

# --------- Mouse callback for point selection ---------
def click_event(event, x, y, flags, param):
    global clicked_points, display_frame
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        cv2.circle(display_frame, (x, y), 5, (0, 255, 0), -1)
        if len(clicked_points) == 2:
            cv2.line(display_frame, clicked_points[0], clicked_points[1], (0, 0, 255), 2)
        cv2.imshow("Select Two Points", display_frame)

# --------- Helper: Compute intersection of a line with a horizontal line (y = Y) ---------
def intersect_horizontal(p1, p2, Y):
    x1, y1 = p1
    x2, y2 = p2
    if y2 == y1:
        return None
    t = (Y - y1) / (y2 - y1)
    x = x1 + t * (x2 - x1)
    return (x, Y)

# --------- Helper: Compute intersection of a line with a vertical line (x = X) ---------
def intersect_vertical(p1, p2, X):
    x1, y1 = p1
    x2, y2 = p2
    if x2 == x1:
        return None
    t = (X - x1) / (x2 - x1)
    y = y1 + t * (y2 - y1)
    return (X, y)

# --------- Helper: Compute polygon area using contourArea ---------
def polygon_area(poly):
    if len(poly) < 3:
        return 0
    return cv2.contourArea(np.array(poly, dtype=np.float32))

# --------- Helper: Order 4 points (top-left, top-right, bottom-right, bottom-left) ---------
def order_points(pts):
    pts = np.array(pts, dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    ordered = np.zeros((4, 2), dtype="float32")
    ordered[0] = pts[np.argmin(s)]  # top-left
    ordered[2] = pts[np.argmax(s)]  # bottom-right
    ordered[1] = pts[np.argmin(diff)]  # top-right
    ordered[3] = pts[np.argmax(diff)]  # bottom-left
    return ordered

# --------- Helper: Transform bbox coordinates using homography matrix ---------
def transform_bbox(bbox, M):
    # Convert bbox to four corner points
    x1, y1, x2, y2 = bbox
    corners = np.array([
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2]
    ], dtype="float32")

    # Transform corners using homography matrix
    transformed_corners = cv2.perspectiveTransform(np.array([corners]), M)[0]

    # Compute new bbox from transformed corners
    new_x1 = min(transformed_corners[:, 0])
    new_y1 = min(transformed_corners[:, 1])
    new_x2 = max(transformed_corners[:, 0])
    new_y2 = max(transformed_corners[:, 1])

    return [new_x1, new_y1, new_x2, new_y2]

# --------- Main Processing Function ---------
def process_video(input_video, output_video):
    global clicked_points, display_frame, M_left, M_right

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Error opening video file: {input_video}")
        return

    ret, first_frame = cap.read()
    if not ret:
        print(f"Error reading first frame from: {input_video}")
        return

    orig_h, orig_w = first_frame.shape[:2]

    # Resize first frame for display & point selection.
    display_frame = cv2.resize(first_frame, (DISPLAY_SIZE[1], DISPLAY_SIZE[0]))  # (width, height)
    cv2.namedWindow("Select Two Points", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select Two Points", DISPLAY_SIZE[1], DISPLAY_SIZE[0])
    cv2.setMouseCallback("Select Two Points", click_event)

    print(f"Click two points on the window to define the cut line for {input_video}. Press 'q' when done.")
    while True:
        cv2.imshow("Select Two Points", display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or len(clicked_points) >= 2:
            break
        if key == ord('r'):  # Press 'r' to reselect points
            clicked_points = []  # Reset clicked points
            cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)  # Go back to the START_FRAME
            ret, last_frame = cap.read()
            if ret:
                display_frame = cv2.resize(last_frame, (DISPLAY_SIZE[1], DISPLAY_SIZE[0]))  # Show the last frame
    cv2.destroyWindow("Select Two Points")
    if len(clicked_points) < 2:
        print("Two points were not selected.")
        return

    # Scale selected points back to original resolution.
    scale_x = orig_w / DISPLAY_SIZE[1]
    scale_y = orig_h / DISPLAY_SIZE[0]
    p1 = (int(clicked_points[0][0] * scale_x), int(clicked_points[0][1] * scale_y))
    p2 = (int(clicked_points[1][0] * scale_x), int(clicked_points[1][1] * scale_y))
    print(f"Selected points (original resolution): {p1}, {p2}")

    # Determine orientation of the cut line.
    delta_x = abs(p2[0] - p1[0])
    delta_y = abs(p2[1] - p1[1])

    candidate_quads = []

    if delta_y >= delta_x:
        # Nearly vertical cut: use intersections with top and bottom borders.
        top_int = intersect_horizontal(p1, p2, 0)
        bot_int = intersect_horizontal(p1, p2, orig_h)
        if top_int is None or bot_int is None:
            print("Error computing intersections with top/bottom.")
            return
        # Candidate: Left region quad and Right region quad.
        left_quad = [(0, 0), top_int, bot_int, (0, orig_h)]
        right_quad = [top_int, (orig_w, 0), (orig_w, orig_h), bot_int]
        candidate_quads.append(("left", left_quad))
        candidate_quads.append(("right", right_quad))
    else:
        # Nearly horizontal cut: use intersections with left and right borders.
        left_int = intersect_vertical(p1, p2, 0)
        right_int = intersect_vertical(p1, p2, orig_w)
        if left_int is None or right_int is None:
            print("Error computing intersections with left/right.")
            return
        top_quad = [(0, 0), (orig_w, 0), right_int, left_int]
        bottom_quad = [left_int, right_int, (orig_w, orig_h), (0, orig_h)]
        candidate_quads.append(("top", top_quad))
        candidate_quads.append(("bottom", bottom_quad))

    # Compute areas of candidate quadrilaterals.
    areas = [(name, polygon_area(quad), quad) for name, quad in candidate_quads]
    for name, area, quad in areas:
        print(f"Candidate region '{name}' area: {area}")

    # Choose the quadrilateral with the larger area.
    chosen_name, chosen_area, chosen_quad = max(areas, key=lambda x: x[1])
    print(f"Selected region: {chosen_name} with area {chosen_area}")

    # Force it into a quadrilateral by taking the borders:
    if delta_y >= delta_x:
        if chosen_name == "left":
            new_quad = [(0, 0), (int(chosen_quad[1][0]), 0),
                        (int(chosen_quad[2][0]), orig_h), (0, orig_h)]
        else:
            new_quad = [(int(chosen_quad[0][0]), 0), (orig_w, 0),
                        (orig_w, orig_h), (int(chosen_quad[3][0]), orig_h)]
    else:
        if chosen_name == "top":
            new_quad = [(0, 0), (orig_w, 0),
                        (orig_w, int(right_int[1])), (0, int(left_int[1]))]
        else:
            new_quad = [(0, int(left_int[1])), (orig_w, int(right_int[1])),
                        (orig_w, orig_h), (0, orig_h)]
    # Use the new quadrilateral for homography.
    quad_points = order_points(new_quad)
    print("Final quadrilateral for homography (ordered):")
    print(quad_points)

    # Compute destination rectangle size.
    widthA = np.linalg.norm(quad_points[2] - quad_points[3])
    widthB = np.linalg.norm(quad_points[1] - quad_points[0])
    maxWidth = int(max(widthA, widthB))
    heightA = np.linalg.norm(quad_points[1] - quad_points[2])
    heightB = np.linalg.norm(quad_points[0] - quad_points[3])
    maxHeight = int(max(heightA, heightB))
    print(f"Destination size: {maxWidth} x {maxHeight}")

    dst_pts = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")

    # Compute homography.
    M = cv2.getPerspectiveTransform(quad_points, dst_pts)

    # Store the transformation matrix for left or right video
    if input_video == "left5shifted.mp4":
        M_left = M
    elif input_video == "right5.mp4":
        M_right = M

    # Setup output video writer.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, cap.get(cv2.CAP_PROP_FPS), (maxWidth, maxHeight))

    print("Processing video frames...")
    # Start from START_FRAME and process N_FRAMES
    cap.set(cv2.CAP_PROP_POS_FRAMES, START_FRAME)

    for _ in tqdm(range(N_FRAMES), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break
        warped = cv2.warpPerspective(frame, M, (maxWidth, maxHeight), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        out.write(warped)
    cap.release()
    out.release()
    print(f"Output video saved as {output_video}")

# --------- Process JSON file ---------
def process_json(input_json, output_json, M_left, M_right):
    with open(input_json, "r") as f:
        data = json.load(f)

    for frame in data["frames"]:
        for obj in frame["obj"]:
            color = obj["color"].lower()
            bbox = obj["bbox"]

            # Determine which transformation matrix to use
            if color in ["blue", "purple"]:
                M = M_left
            elif color in ["red", "orange", "yellow"]:
                M = M_right
            else:
                continue  # Skip if color is not in the specified categories

            # Transform bbox coordinates
            if M is not None:
                obj["bbox"] = transform_bbox(bbox, M)

    # Convert numpy.float32 to float for JSON serialization
    def convert_floats(obj):
        if isinstance(obj, np.float32):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_floats(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_floats(item) for item in obj]
        return obj

    data = convert_floats(data)

    # Save the updated JSON
    with open(output_json, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Updated JSON saved as {output_json}")

if __name__ == "__main__":
    # Process videos and compute transformation matrices
    for input_video, output_video in zip(INPUT_VIDEOS, OUTPUT_VIDEOS):
        clicked_points = []  # Reset clicked points for each video
        process_video(input_video, output_video)

    # Process JSON file
    process_json(INPUT_JSON, OUTPUT_JSON, M_left, M_right)
