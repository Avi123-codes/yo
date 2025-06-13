import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set up device for GPU acceleration if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

class Model(nn.Module):
    def __init__(self, num_classes=26):
        super(Model, self).__init__()
        self.features = nn.Sequential(
            # Layer 1
            nn.Conv2d(in_channels = 1, out_channels =32, kernel_size = 3, padding =1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            # Layer 2
            nn.Conv2d(32, 64, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            # Layer 3
            nn.Conv2d(64, 128, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            # Layer 4
            nn.Conv2d(128, 256, kernel_size = 3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
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

torch.manual_seed(20)

# Instantiate the model and move it to the selected device
model = Model().to(device)
print(model)

# Load the dataset
url = 'https://raw.githubusercontent.com/Avi123-codes/yo/refs/heads/main/asl_image_dataset%20(1).csv  '
data = pd.read_csv(url)

# ---Data Processing___
# Encode species using .map() - good for fixed, known categories
letters_mapping = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4,
    "F": 5,
    "G": 6,
    "H": 7,
    "I": 8,
    "J": 9,
    "K": 10,
    "L": 11,
    "M": 12,
    "N": 13,
    "O": 14,
    "P": 15,
    "Q": 16,
    "R": 17,
    "S": 18,
    "T": 19,
    "U": 20,
    "V": 21,
    "W": 22,
    "X": 23,
    "Y": 24,
    "Z": 25
}
data["label"] = data["label"].map(letters_mapping)

# Split input (X) and output (y)
X = data.drop("label", axis = 1).values
y = data["label"].values

# Split data into training and testing sets
X_train = X
y_train = y

# Convert NumPy arrays to PyTorch Tensors and move to device
X_train = torch.FloatTensor(X_train).to(device)
X_train = X_train.view(-1, 1, 64, 64)
y_train = torch.LongTensor(y_train).to(device)

# Define Loss function and Optimizer
criteria = nn.CrossEntropyLoss().to(device) 
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)


# ---Training Loop---
epochs = 500
losses = []

for i in range(epochs):
    # Forward Pass: Compute predicted y by passing X to model
    y_pred = model.forward(X_train)

    # Compute and print loss
    loss = criteria(y_pred, y_train)

    # Store loss (detach from graph to prevent memory leak)
    losses.append(loss.item())

    # Print loss every 50 epochs
    if i % 50 == 0:
        print(f"Epoch: {i}, Loss: {loss.item():.4f}") 

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad() 
    loss.backward()     
    optimizer.step()      

print(f"\nFinal training Loss: {losses[-1]:.4f}")

# --- Plotting the Loss ---
plt.figure(figsize=(10, 6))
plt.plot(range(epochs), losses)
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.title("Training Loss Over Epochs")
plt.grid(True)
plt.show()

# --- A few test predictions ---
print("\n--- Sample Test Predictions ---")
for i in range(10):
    actual_label = y_test[i].item()
    predicted_label = predicted_classes[i].item()
    print(f"Sample {i+1}: Actual: {actual_label}, Predicted: {predicted_label} {'(Correct)' if actual_label == predicted_label else '(Incorrect)'}")

#Save model
torch.save(model.state_dict(), "ASL_model.pt")
