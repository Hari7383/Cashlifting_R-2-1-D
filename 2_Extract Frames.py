import cv2
import os

def extract_frames(video_path, output_dir, every_n_frames=5):
    cap = cv2.VideoCapture(video_path)
    os.makedirs(output_dir, exist_ok=True)
    count = 0
    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % every_n_frames == 0:
            cv2.imwrite(f"{output_dir}/frame_{frame_id:04d}.jpg", frame)
            frame_id += 1
        count += 1
    cap.release()

# Apply to all videos
import glob

classes = ['cashlifting', 'normal']
for cls in classes:
    video_paths = glob.glob(f"dataset/{cls}/*.mp4")
    for vid in video_paths:
        name = os.path.splitext(os.path.basename(vid))[0]
        extract_frames(vid, f"processed_frames/{cls}/{name}")
