# import os
# from PIL import Image
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from torchvision.models.video import r3d_18

# # Video Frame Dataset
# class VideoFrameDataset(Dataset):
#     def __init__(self, root, classes, num_frames=32):
#         self.samples = []
#         self.classes = classes
#         self.num_frames = num_frames
#         self.transform = transforms.Compose([
#             transforms.Resize((112, 112)),
#             transforms.ToTensor(),
#         ])

#         for label, cls in enumerate(classes):
#             class_dir = os.path.join(root, cls)
#             for folder in os.listdir(class_dir):
#                 frame_folder = os.path.join(class_dir, folder)
#                 if os.path.isdir(frame_folder):
#                     frames = sorted(os.listdir(frame_folder))
#                     frame_paths = [os.path.join(frame_folder, f) for f in frames[:num_frames]]
#                     if len(frame_paths) >= num_frames:
#                         self.samples.append((frame_paths, label))

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         frame_paths, label = self.samples[idx]
#         frames = [self.transform(Image.open(f)) for f in frame_paths]
#         video_tensor = torch.stack(frames)  # (T, C, H, W)
#         video_tensor = video_tensor.permute(1, 0, 2, 3)  # (C, T, H, W)
#         return video_tensor, label

# # Setup
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = r3d_18(pretrained=True)
# model.fc = nn.Linear(model.fc.in_features, 2)  # Binary classification
# model = model.to(device)

# # Data
# dataset = VideoFrameDataset("processed_frames", ['cashlifting', 'normal'])
# train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

# # Loss and Optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# # Training Loop
# for epoch in range(100):
#     model.train()
#     running_loss = 0.0
#     for videos, labels in train_loader:
#         videos, labels = videos.to(device), labels.to(device)
#         outputs = model(videos)
#         loss = criterion(outputs, labels)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#     print(f"Epoch [{epoch+1}/100], Loss: {running_loss:.4f}")

# # Save model
# torch.save(model.state_dict(), "cashlifting_model_0.1.pth")
# print("Model saved as 'cashlifting_model.pth'")


import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.models.video import r2plus1d_18
from sklearn.metrics import classification_report
from collections import Counter

class VideoFrameDataset(Dataset):
    def __init__(self, root, classes, num_frames=32):
        self.samples = []
        self.classes = classes
        self.num_frames = num_frames
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomCrop((112, 112)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        for label, cls in enumerate(classes):
            class_dir = os.path.join(root, cls)
            for folder in os.listdir(class_dir):
                frame_folder = os.path.join(class_dir, folder)
                if os.path.isdir(frame_folder):
                    frames = sorted(os.listdir(frame_folder))
                    frame_paths = [os.path.join(frame_folder, f) for f in frames]
                    sampled = self.uniform_sample(frame_paths)
                    if sampled:
                        self.samples.append((sampled, label))

    def uniform_sample(self, frame_paths):
        if len(frame_paths) < self.num_frames:
            return None
        idxs = np.linspace(0, len(frame_paths) - 1, self.num_frames, dtype=int)
        return [frame_paths[i] for i in idxs]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        frame_paths, label = self.samples[idx]
        frames = [self.transform(Image.open(f).convert("RGB")) for f in frame_paths]
        video_tensor = torch.stack(frames)  # (T, C, H, W)
        video_tensor = video_tensor.permute(1, 0, 2, 3)  # (C, T, H, W)
        return video_tensor, label

# =============================
# 2. Model Setup (R2Plus1D_18)
# =============================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = r2plus1d_18(pretrained=True)
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, 2)  # Binary classification
)
model = model.to(device)

# ===================
# 3. Data Preparation
# ===================
classes = ['cashlifting', 'normal']
dataset = VideoFrameDataset("processed_frames", classes)

# Class balancing
label_counts = Counter([label for _, label in dataset.samples])
weights = [1.0 / label_counts[label] for _, label in dataset.samples]
sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
train_loader = DataLoader(dataset, batch_size=4, sampler=sampler)

# ===========================
# 4. Loss, Optimizer, Scheduler
# ===========================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

# ================
# 5. Training Loop
# ================
for epoch in range(30):
    model.train()
    running_loss = 0.0
    for videos, labels in train_loader:
        videos, labels = videos.to(device), labels.to(device)
        outputs = model(videos)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    scheduler.step(avg_loss)
    print(f"Epoch [{epoch+1}/30], Loss: {avg_loss:.4f}")

# ===================
# 6. Evaluation Phase
# ===================
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for videos, labels in train_loader:
        videos = videos.to(device)
        outputs = model(videos)
        preds = torch.argmax(outputs, dim=1).cpu()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())

print("\nClassification Report:")
print(classification_report(all_labels, all_preds, target_names=classes))

# ================
# 7. Save the Model
# ================
torch.save(model.state_dict(), "improved_cashlifting_model_01.pth")
print("\n Model saved as 'improved_cashlifting_model.pth'")
