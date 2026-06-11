
"""
# Step 01 — Conv1D Intuition

## Core Philosophy



Convolution is:

    localized structured information aggregation

From engineering point of view the most popular use case is:

    "applying a filter"

A convolution system allows tensors to interact
with their local neighborhoods.

Instead of observing the entire tensor globally,
convolution observes:

    local interaction regions

This becomes one of the foundational ideas behind:

- CNNs
- audio processing
- signal systems
- transformers (local attention variants)
- diffusion systems
- scientific simulations

---

# Local Interaction Intuition

Suppose we have a 1D tensor:

    x = [2, 4, 6, 8, 10]

And a kernel:

    w = [1, 0, -1]

The kernel acts as:

    a local interaction rule

The kernel slides across the tensor and computes:

    structured local relationships

At each position:

    local neighborhood
        ↓
    weighted interaction
        ↓
    aggregated response

---

# Mathematical Formulation

For a 1D convolution:

y[i] = Σ x[i + k] * w[k]

More formally:

"""
# Mathematical representation
# y[i] = \sum_{k=0}^{K-1} x[i+k] w[k]

"""
Where:

- x = input tensor
- w = kernel tensor
- y = output tensor
- K = kernel size

---

# Geometric Interpretation

The kernel scans through the tensor field.

At each location:

- a local neighborhood is extracted
- local interactions are computed
- an interaction response is generated

Thus convolution becomes:

    local structure
        ↓
    interaction
        ↓
    representation

---

# Important Insight

Convolution does NOT understand meaning.

It only captures:

    local numerical structure

Meaning emerges later through:

- repeated interactions
- hierarchical representations
- deep stacking

---

# In This Step

We will explore:

1. Manual Conv1D intuition
2. Sliding window interactions
3. PyTorch Conv1D
4. Kernel interaction behavior
5. Local aggregation systems
"""
#%%

import torch
import torch.nn as nn

torch.manual_seed(42)

#%% [markdown]
"""
# 1. Creating a Simple 1D Tensor

This tensor can represent:

- signal intensity
- audio amplitude
- sequential measurements
- token interaction strengths

We intentionally use a small tensor
to visualize local interactions clearly.
"""
#%%

x = torch.tensor([2., 4., 6., 8., 10.])

print("Input Tensor:")
print(x)

print("\nTensor Shape:")
print(x.shape)

#%% [markdown]
"""
# 2. Creating a Kernel

The kernel defines:

    local interaction behavior

Kernel:

    [1, 0, -1]

Interaction meaning:

- compare left region
- ignore center
- compare right region

This kernel becomes sensitive to:

    local directional changes
"""
#%%

kernel = torch.tensor([1., 0., -1.])

print("Kernel:")
print(kernel)

#%% [markdown]
"""
# 3. Manual Sliding Window Convolution

We now manually simulate convolution.

Input:

    [2, 4, 6, 8, 10]

Kernel size:

    3

Sliding windows:

Window 1:
    [2, 4, 6]

Window 2:
    [4, 6, 8]

Window 3:
    [6, 8, 10]

At each step:

    local neighborhood
        ↓
    weighted interaction
        ↓
    aggregation
"""
#%%

output = []

kernel_size = kernel.shape[0]

for i in range(len(x) - kernel_size + 1):

    # Extract local neighborhood
    window = x[i:i + kernel_size]

    # Local interaction
    interaction = window * kernel

    # Aggregation
    response = interaction.sum()

    print(f"\nPosition {i}")
    print("-------------------")

    print("Window:")
    print(window)

    print("Interaction:")
    print(interaction)

    print("Aggregated Response:")
    print(response)

    output.append(response)

output = torch.tensor(output)

#%% [markdown]
"""
# 4. Final Output Tensor

The output tensor contains:

    local interaction responses

Each value represents:

    how strongly the local region
    responds to the kernel rule
"""
#%%

print("\nFinal Output Tensor:")
print(output)

#%% [markdown]
"""
# 5. Understanding Interaction Geometry

Observe what happened:

Input Tensor:

    [2, 4, 6, 8, 10]

Kernel scans locally:

    [2,4,6]
    [4,6,8]
    [6,8,10]

The convolution system never observes
the full tensor simultaneously.

It only observes:

    localized interaction fields

This locality is extremely important.

Modern deep learning systems often build:

    global intelligence
from:
    local interactions
"""
#%% [markdown]
"""
# 6. PyTorch Conv1D

PyTorch expects Conv1D input shape:

    [batch, channels, length]

Why?

Because Conv1D operates over:

    sequential tensor fields
"""
#%%

x_conv = x.reshape(1, 1, -1)

print("Conv1D Input Shape:")
print(x_conv.shape)

#%% [markdown]
"""
# 7. Creating Conv1D Layer

We create:

- 1 input channel
- 1 output channel
- kernel size = 3

The layer learns interaction rules.
"""
#%%

conv1d = nn.Conv1d(
    in_channels=1,
    out_channels=1,
    kernel_size=3,
    bias=False
)

#%% [markdown]
"""
# 8. Manually Assigning Kernel Weights

Instead of random weights,
we manually assign:

    [1, 0, -1]

This allows us to compare
manual convolution with PyTorch convolution.
"""
#%%

with torch.no_grad():

    conv1d.weight[:] = torch.tensor([[[1., 0., -1.]]])

print("Conv1D Weights:")
print(conv1d.weight)

#%% [markdown]
"""
# 9. Running Conv1D

PyTorch now performs:

    structured local interaction aggregation

automatically.
"""
#%%

y = conv1d(x_conv)

print("Conv1D Output:")
print(y)

#%% [markdown]
"""
# 10. Removing Extra Dimensions

Current shape:

    [batch, channels, length]

We simplify for visualization.
"""
#%%

y_squeezed = y.squeeze()

print("Simplified Output:")
print(y_squeezed)

#%% [markdown]
"""
# 11. Comparing Manual vs PyTorch Outputs
"""
#%%

print("Manual Output:")
print(output)

print("\nPyTorch Output:")
print(y_squeezed)

#%% [markdown]
"""
# Final Conceptual Understanding

Conv1D is fundamentally:

    localized tensor interaction

A kernel represents:

    an interaction rule

Convolution performs:

    neighborhood aggregation

Feature maps become:

    interaction response fields

This is the beginning of:

- spatial reasoning
- representation emergence
- hierarchical feature learning
- tensor field systems

---

# Deep Insight

One of the deepest ideas in modern AI:

    Local interactions can create
    global representations.

Convolution systems are one of the first
major demonstrations of this principle.
"""