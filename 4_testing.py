import cv2
import os
from PIL import Image
import torch
from torchvision import transforms
from torchvision.models.video import r3d_18
import torch.nn as nn
from torchvision.models.video import R3D_18_Weights


# Same transform as training
transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
])

def extract_temp_frames(video_path, temp_dir="temp_frames", every_n=5):
    os.makedirs(temp_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    saved_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % every_n == 0:
            frame_path = os.path.join(temp_dir, f"frame_{saved_idx:03d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_idx += 1
        frame_idx += 1
    cap.release()
    return sorted(os.listdir(temp_dir))[:16]

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_video(video_path, model_path="cashlifting_model_0.1.pth"):
    temp_dir = "temp_frames"
    frame_names = extract_temp_frames(video_path, temp_dir=temp_dir)
    frames = [transform(Image.open(os.path.join(temp_dir, f))) for f in frame_names]
    video_tensor = torch.stack(frames).permute(1, 0, 2, 3).unsqueeze(0).to(device)

    model = r3d_18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        output = model(video_tensor)
        pred = torch.argmax(output, dim=1).item()

    return ['cashlifting', 'normal'][pred]

# Example usage:
print(predict_video("dataset/cashlifting/v1.mp4"))
