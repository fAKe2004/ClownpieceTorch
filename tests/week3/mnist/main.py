# %% [markdown]
# # MNIST Handwritten Digit Classification
# 
# This notebook demonstrates a classic classification task using the `clownpiece` library to recognize handwritten digits from the MNIST dataset.
# 
# We will:
# 1.  Load the MNIST dataset using `torchvision`.
# 2.  Preprocess the data and convert it to `clownpiece` Tensors.
# 3.  Build and train a simple MLP classifier.
# 4.  Use `CrossEntropyLoss` for training.
# 5.  Evaluate the model's accuracy on a test set.

# %%
import sys
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('../../../')

import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch

import clownpiece
from clownpiece import Tensor
from clownpiece.autograd import no_grad
from clownpiece.nn import Module, Linear, ReLU, Sequential, CrossEntropyLoss, MultiheadAttention, LayerNorm

# %%
# Load MNIST dataset using torchvision
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

print("MNIST dataset loaded successfully.")

# %%
# Define the model
input_features = 784  # 28x28 images flattened
num_classes = 10

hidden_dim = 32
kernel_size = 4

class ImageEmbedding(Module):
    def __init__(self, input_features, hidden_dim, kernel_size = 4):
        super().__init__()
        self.input_features = input_features
        self.patch_size = kernel_size * kernel_size
        
        assert input_features % self.patch_size == 0, "input_features must be divisible by patch_size"
        self.num_patches = self.input_features // self.patch_size
        
        self.linear = Linear(self.patch_size, hidden_dim)

    def forward(self, x):
        # x: (batch_size, input_features)
        # Reshape to (batch_size, num_patches, patch_size)
        patches = x.reshape((-1, self.num_patches, self.patch_size))
        
        # Project patches to embeddings
        return self.linear(patches) # (batch_size, num_patches, patch_size)
    
class TransformerBlock(Module):
    def __init__(self, hidden_dim, num_heads, ffn_dim):
        super().__init__()
        self.attention = MultiheadAttention(hidden_dim, num_heads, True)
        
        self.mlp = Sequential(
            Linear(hidden_dim, ffn_dim),
            ReLU(),
            Linear(ffn_dim, hidden_dim)
        )

        self.layer_norm1 = LayerNorm(hidden_dim)
        self.layer_norm2 = LayerNorm(hidden_dim)

    def forward(self, x):
        x = x + self.attention(x)
        x = self.layer_norm1(x)
        x = x + self.mlp(x)
        x = self.layer_norm2(x)
        return x
    
class Reduce(Module):
    def forward(self, x):
        return x.mean(dim=-1)

model = Sequential(
    ImageEmbedding(input_features=input_features, hidden_dim=hidden_dim, kernel_size=kernel_size),
    ReLU(),
    TransformerBlock(hidden_dim, num_heads=4, ffn_dim=2*hidden_dim),
    ReLU(),
    Reduce(),
    Linear(input_features // (kernel_size * kernel_size), num_classes)
)

print("Model Architecture:")
print(model)

# %%
# Loss and training parameters
loss_fn = CrossEntropyLoss()
learning_rate = 1e-4
epochs = 10
train_losses = []
test_accuracies = []

# %%
# %%
# Training loop
for epoch in range(epochs):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        # Flatten the data and convert to clownpiece Tensors
        data_flat = data.view(data.shape[0], -1)
        X = Tensor(data_flat.numpy().tolist())
        y = Tensor(target.numpy().tolist(), requires_grad=False)

        # Forward pass
        logits = model(X)
        # print(logits)

        # Calculate loss
        loss = loss_fn(logits, y)

        # Backward pass
        loss.backward()

        # Update weights
        with no_grad():
            for param in model.parameters():
                if param.grad is not None:
                    # print(param.grad)
                    param.copy_(param - param.grad * learning_rate)
        
        # Zero gradients
        for param in model.parameters():
            if param.grad is not None:
                param.grad = None

        if batch_idx % 1 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}]\tLoss: {loss.item():.6f}')
            train_losses.append(loss.item())
    
    # Evaluation on test set
    model.eval()
    correct = 0
    with no_grad():
        for data, target in test_loader:
            data_flat = data.view(data.shape[0], -1)
            X_test = Tensor(data_flat.numpy().tolist())
            
            logits = model(X_test)
            pred = np.argmax(np.array(logits.tolist()), axis=1)
            correct += np.sum(pred == target.numpy())
    
    accuracy = 100. * correct / len(test_loader.dataset)
    test_accuracies.append(accuracy)
    print(f'\nTest set: Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')

# %%
# Plotting results
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(train_losses)
ax1.set_title("Training Loss")
ax1.set_xlabel("Iterations (x100)")
ax1.set_ylabel("Cross-Entropy Loss")
ax1.grid(True)

ax2.plot(test_accuracies)
ax2.set_title("Test Accuracy")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.grid(True)

plt.show()


