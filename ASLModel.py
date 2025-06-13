%env CUDA_LAUNCH_BLOCKING=1

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import TensorDataset, DataLoader
import os
import json
from google.colab import drive
from torch.utils.data import TensorDataset, DataLoader

drive.mount('/content/drive')

# Set device
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using Device: {device}.")

# --- Define Model ---
class Model(nn.Module):
    def __init__(self, num_classes=26):
        super(Model, self).__init__()
        self.features = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Layer 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Layer 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Layer 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# --- Load and Preprocess Data ---
letters_mapping = {
    "A": 0, "B": 1, "C": 2, "D": 3, "E": 4,
    "F": 5, "G": 6, "H": 7, "I": 8, "J": 9,
    "K":10, "L":11, "M":12, "N":13, "O":14,
    "P":15,"Q":16, "R":17, "S":18, "T":19,
    "U":20,"V":21, "W":22, "X":23, "Y":24, "Z":25
}

url = '/content/drive/MyDrive/asl_alphabet_full.csv'
data = pd.read_csv(url)

# Map labels safely
data["label"] = data["label"].str.upper()
data["label"] = data["label"].map(letters_mapping)
data = data.dropna(subset=['label']).copy()
data['label'] = data['label'].astype(int)

# Split input (X) and output (y)
X = data.drop("label", axis=1).values.astype("float32")
y_encoded = data["label"].values

# Normalize inputs
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train/Validation Split
X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train_np).to(device).view(-1, 1, 64, 64)
y_train = torch.LongTensor(y_train_np).to(device)
X_val = torch.FloatTensor(X_val_np).to(device).view(-1, 1, 64, 64)
y_val = torch.LongTensor(y_val_np).to(device)

# Create datasets and loaders
batch_size = 64
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Print for debugging
print(f"Labels min: {y_train.min().item()}, max: {y_train.max().item()}")

# Instantiate model
model = Model().to(device)
print(model)

# Loss and optimizer
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.003)

# Optional LR scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# Accuracy function
def calculate_accuracy(loader, model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Training loop
epochs = 15
losses = []

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()

    train_acc = calculate_accuracy(train_loader, model)
    val_acc = calculate_accuracy(val_loader, model)

    avg_loss = running_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    losses.append(avg_loss)

    print(f"Epoch {epoch+1}/{epochs}, "
          f"Train Loss: {avg_loss:.4f}, "
          f"Val Loss: {avg_val_loss:.4f}, "
          f"Train Acc: {train_acc:.2f}%, "
          f"Val Acc: {val_acc:.2f}%")

    scheduler.step()

# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(range(epochs), losses)
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.title("Training Loss Over Epochs")
plt.grid(True)
plt.show()

# Save model and class mapping
torch.save(model.state_dict(), "ASL_model.pt")
idx_to_letter = {v: k for k, v in letters_mapping.items()}
with open("class_mapping.json", "w") as f:
    json.dump(idx_to_letter, f)

print("Model and class mapping saved.")
