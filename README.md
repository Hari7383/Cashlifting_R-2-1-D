3_Torchvision_Model_Traine Explanation :

1. Custom Dataset Class: VideoFrameDataset
This class loads sequences of frames from folders — each folder represents one video clip.

Key steps:
root: path to processed frame folders (e.g., processed_frames/cashlifting/vid1/, etc.).

classes: class names (e.g., ['cashlifting', 'normal']).

num_frames: how many frames to sample from each video.

Transforms:

Resize → RandomCrop → HorizontalFlip → ToTensor

Sampling:

Uniformly sample num_frames (default = 32) frames per folder to ensure consistency.
____________________________________________________________________________________________________________________________
2. Model: R2Plus1D-18
It’s a 3D convolutional video model from torchvision.

Replace the last fully connected layer (model.fc) to match your classification task:

python
Copy
Edit
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(model.fc.in_features, 2)  # binary: cashlifting or normal
)
____________________________________________________________________________________________________________________________
3. Dataset & DataLoader
Initialize the dataset with all labeled video frame folders.

To handle class imbalance, you use WeightedRandomSampler so the model sees balanced samples of both classes during training.
____________________________________________________________________________________________________________________________
4. Loss, Optimizer, Scheduler
Loss: CrossEntropyLoss (standard for classification).

Optimizer: Adam with LR = 1e-4.

Scheduler: ReduceLROnPlateau, which reduces the learning rate if the loss doesn’t improve for a few epochs.
____________________________________________________________________________________________________________________________
5. Training Loop
Loops through the dataset for 30 epochs.

For each batch:

Moves input to GPU if available.

Computes forward pass → loss → backpropagation → updates weights.

Tracks average loss.

Scheduler adjusts the LR if loss plateaus.
____________________________________________________________________________________________________________________________
6. Evaluation
After training, it evaluates on the same dataset (you can replace this with a validation/test set).

It predicts labels and compares against ground truth.

Uses sklearn.metrics.classification_report() to print precision, recall, F1-score per class.
____________________________________________________________________________________________________________________________
7. Save Model
Saves trained model to disk:

python
Copy
Edit
torch.save(model.state_dict(), "improved_cashlifting_model_01.pth")
