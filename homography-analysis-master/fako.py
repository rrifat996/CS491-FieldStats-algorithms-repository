import cv2

def crop_and_save_video(input_video, output_video, start_sec, end_sec, crop_x, crop_y):
    cap = cv2.VideoCapture(input_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    if crop_x[1] > width or crop_y[1] > height:
        print("Crop dimensions exceed video size")
        return
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (crop_x[1] - crop_x[0], crop_y[1] - crop_y[0]))
    
    start_frame = start_sec * fps
    end_frame = end_sec * fps
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if current_frame > end_frame:
            break
        
        cropped_frame = frame[crop_y[0]:crop_y[1], crop_x[0]:crop_x[1]]
        out.write(cropped_frame)
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Video processing complete.")

# Usage
crop_and_save_video(
    input_video='right5.mp4', 
    output_video='aaaaaaa.mp4', 
    start_sec=30, 
    end_sec=60, 
    crop_x=(400, 500), 
    crop_y=(700, 800)
)