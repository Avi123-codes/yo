import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Set up device for GPU acceleration if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

class Model(nn.Module):
    def __init__(self, input_features = 4, h1 = 8, h2 = 8, output_features = 3):
        super().__init__()
        self.fc1 = nn.Linear(input_features, h1) # nn.Sequential for a CNN model
        self.fc2 = nn.Linear(h1, h2)
        self.out = nn.Linear(h2, output_features)

    def forward(self, X):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.out(X)
        return X

torch.manual_seed(20)

# Instantiate the model and move it to the selected device
model = Model().to(device)
print(model)

# Load the dataset
url = 'https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv'
data = pd.read_csv(url)

# ---Data Processing___
# Encode species using .map() - good for fixed, known categories
species_mapping = {
    "setosa": 0,
    "versicolor": 1,
    "virginica": 2
}
data["species"] = data["species"].map(species_mapping)

# Split input (X) and output (y)
X = data.drop("species", axis = 1).values # Convert to NumPy array directly
y = data["species"].values # Convert to NumPy array directly

# Split data into training and testing sets
# Correct Order: Output order is X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 42)

# Convert NumPy arrays to PyTorch Tensors and move to device
X_train = torch.FloatTensor(X_train).to(device)
y_train = torch.LongTensor(y_train).to(device) # Labels should be LongTensor

X_test = torch.FloatTensor(X_test).to(device)
y_test = torch.LongTensor(y_test).to(device)

# Define Loss function and Optimizer
criteria = nn.CrossEntropyLoss().to(device) # Move loss function to device too
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)


# ---Training Loop---
epochs = 500
losses = [] # To store training loss values

for i in range(epochs):
    # Forward Pass: Compute predicted y by passing X to model
    y_pred = model.forward(X_train)

    # Compute and print loss
    loss = criteria(y_pred, y_train)

    # Store loss (detach from graph to prevent memory leak)
    losses.append(loss.item())

    # Print loss every 50 epochs
    if i % 50 == 0:
        print(f"Epoch: {i}, Loss: {loss.item():.4f}") # Use .item() for printing

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad() # Clear previous gradients
    loss.backward()       # Compute gradients
    optimizer.step()      # Update model parameters

print(f"\nFinal training Loss: {losses[-1]:.4f}")


# ---Model Evaluation (on Test Set) ---
print("\n--- Model Evaluation ---")
with torch.no_grad(): # Disable gradient calculation for inference
    model.eval() # Set model to evaluation mode (e.g., disables dropout if used)
    y_eval = model(X_test) # Use model() instead of model.forward() for consistency
    test_loss = criteria(y_eval, y_test)
    print(f"Final Test Loss: {test_loss.item():.4f}")

    # Get predictions: choose the class with the highest probability
    predicted_classes = torch.argmax(y_eval, dim=1)

    # Calculate accuracy
    accuracy = (predicted_classes == y_test).sum().item() / len(y_test)
    print(f"Test Accuracy: {accuracy:.4f}")

# --- Plotting the Loss ---
plt.figure(figsize=(10, 6))
plt.plot(range(epochs), losses)
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.title("Training Loss Over Epochs")
plt.grid(True)
plt.show()

# --- Optional: See a few test predictions ---
print("\n--- Sample Test Predictions ---")
for i in range(10): # Look at first 10 test samples
    actual_label = y_test[i].item()
    predicted_label = predicted_classes[i].item()
    print(f"Sample {i+1}: Actual: {actual_label}, Predicted: {predicted_label} {'(Correct)' if actual_label == predicted_label else '(Incorrect)'}")


#Save model
torch.save(model.state_dict(), "iris_model.pt")
