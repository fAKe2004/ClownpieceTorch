import torch

# Create tensors with requires_grad=True to track computations
x = torch.tensor([2.0], requires_grad=True)
y = torch.tensor([3.0], requires_grad=True)

# Define a simple computation
z = x * y + y**2  # z = 2*3 + 3^2 = 6 + 9 = 15

# Backward pass: compute gradients of z w.r.t. x and y
z.backward()

# Print gradients
print(f"x.grad = {x.grad}")  # ∂z/∂x = y = 3.0
print(f"y.grad = {y.grad}")  # ∂z/∂y = x + 2*y = 2 + 6 = 8.0