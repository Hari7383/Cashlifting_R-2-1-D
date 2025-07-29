# import os
# import cv2
# import torch
# import torch.nn as nn
# import numpy as np
# from PIL import Image
# import torchvision.transforms as transforms
# from torchvision.models.video import r3d_18

# # Device config
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load model
# model = r3d_18(weights=None)
# model.fc = nn.Linear(model.fc.in_features, 2)
# model.load_state_dict(torch.load("improved_cashlifting_model.pth", map_location=device))
# model = model.to(device)
# model.eval()

# # Transform
# transform = transforms.Compose([
#     transforms.Resize((112, 112)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.43216, 0.394666, 0.37645],
#                          [0.22803, 0.22145, 0.216989])
# ])

# # Prediction function for a clip of 16 frames
# def predict_clip(frames):
#     with torch.no_grad():
#         frames_tensor = [transform(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))) for f in frames]
#         frames_tensor = torch.stack(frames_tensor)  # [T, C, H, W]
#         frames_tensor = frames_tensor.permute(1, 0, 2, 3).unsqueeze(0).to(device)  # [1, C, T, H, W]

#         output = model(frames_tensor)
#         probs = torch.softmax(output, dim=1)
#         _, pred = torch.max(probs, 1)
#         label = 'cashlifting' if pred.item() == 1 else 'normal'
#         confidence = probs[0, pred.item()].item()
#         return label, confidence

# # Annotate entire video based on any clip being cashlifting
# def annotate_video(video_path):
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         print("Failed to open video.")
#         return

#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     frames = []
#     window_size = 16
#     stride = 8

#     detected_cashlifting = False
#     all_frames = []

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         all_frames.append(frame)
#         frames.append(frame)

#         if len(frames) == window_size:
#             label, conf = predict_clip(frames)
#             if label == "cashlifting":
#                 detected_cashlifting = True
#                 print(f"[!] CASHLIFTING detected with confidence {conf:.2f}")

#             # Slide window
#             frames = frames[stride:]

#     # Annotate final result on each frame
#     final_label = "CASHLIFTING" if detected_cashlifting else "NORMAL"
#     color = (0, 0, 255) if detected_cashlifting else (0, 255, 0)

#     for frame in all_frames:
#         cv2.putText(frame, final_label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
#         cv2.imshow("Video Classification", frame)
#         if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# # Run it
# annotate_video("dataset/cashlifting/v2.mp4")

import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models.video import r2plus1d_18  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = r2plus1d_18(weights=None)  
in_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(in_features, 2)
)

model.load_state_dict(torch.load("improved_cashlifting_model.pth", map_location=device))
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.43216, 0.394666, 0.37645],
                         [0.22803, 0.22145, 0.216989])
])


def predict_clip(frames):
    with torch.no_grad():
        frames_tensor = [transform(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))) for f in frames]
        frames_tensor = torch.stack(frames_tensor)  # [T, C, H, W]
        frames_tensor = frames_tensor.permute(1, 0, 2, 3).unsqueeze(0).to(device)  # [1, C, T, H, W]

        output = model(frames_tensor)
        probs = torch.softmax(output, dim=1)
        _, pred = torch.max(probs, 1)
        label = 'cashlifting' if pred.item() == 1 else 'normal'
        confidence = probs[0, pred.item()].item()
        return label, confidence

def annotate_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Failed to open video.")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frames = []
    window_size = 16
    stride = 8

    detected_cashlifting = False
    all_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        all_frames.append(frame)
        frames.append(frame)

        if len(frames) == window_size:
            label, conf = predict_clip(frames)
            if label == "cashlifting":
                detected_cashlifting = True
                print(f"[!] CASHLIFTING detected with confidence {conf:.2f}")

            # Slide window
            frames = frames[stride:]

    # Annotate final result on each frame
    final_label = "CASHLIFTING" if detected_cashlifting else "NORMAL"
    color = (0, 0, 255) if detected_cashlifting else (0, 255, 0)

    for frame in all_frames:
        cv2.putText(frame, final_label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.imshow("Video Classification", frame)
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run it
if __name__ == "__main__":
    video_path = "dataset/cashlifting/v2.mp4" 
    annotate_video(video_path)
