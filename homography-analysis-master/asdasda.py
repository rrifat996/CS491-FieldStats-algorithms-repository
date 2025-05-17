import cv2
import numpy as np

# Parameters (adjustable)
alpha = 0.05  # Background updating rate
threshold_factor = 2.5  # Standard deviation multiplier for segmentation

# Open video file or webcam
cap = cv2.VideoCapture('aa.mp4')  # replace with 0 for webcam

# Initialization
ret, frame = cap.read()
if not ret:
    raise ValueError("Failed to read the video file.")
 
# Convert first frame to grayscale
frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype('float32')

# Initialize mean and standard deviation for background model
background_mean = frame_gray.copy()
background_std = np.full(frame_gray.shape, 10.0, dtype='float32')  # initial arbitrary std dev

while True:
    ret, frame = cv2.VideoCapture.read(cap)
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype('float32')

    # Compute difference between current frame and background
    diff = cv2.absdiff(frame_gray, background_mean)

    # Detect foreground pixels
    foreground_mask = diff > (threshold_factor * background_std)
    foreground_mask = foreground_mask.astype('uint8') * 255

    # Update the background model only for static pixels
    static_pixels = diff <= (threshold_factor * background_std)
    background_mean[static_pixels] = alpha * frame_gray[static_pixels] + (1 - alpha) * background_mean[static_pixels]
    background_std[static_pixels] = alpha * diff[static_pixels] + (1 - alpha) * background_std[static_pixels]

    # Apply morphological operations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))  
    foreground_mask = cv2.morphologyEx(foreground_mask, cv2.MORPH_OPEN, kernel=kernel)

    # Visualize results
    cv2.imshow('Original', frame)
    cv2.imshow('Foreground Mask', foreground_mask)

    # Break loop if 'q' pressed
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()