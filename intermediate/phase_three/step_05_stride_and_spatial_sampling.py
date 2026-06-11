#%% [markdown]
"""
# Step 05 — Stride and Spatial Sampling

# Core Philosophy

Stride is not merely:

    "how much the kernel moves"

Stride controls:

    spatial sampling density

A convolution system cannot observe every possible
local interaction at all computational scales.

Stride determines:

    how densely the tensor field is sampled

This introduces one of the most important ideas in
deep learning systems:

    representation-resolution tradeoff

---

# Important Intuition

Small stride:

    dense observation

Large stride:

    coarse observation

Thus stride controls:

- spatial detail
- computational cost
- interaction granularity
- information preservation

---

# Geometric Interpretation

A convolution kernel slides through
a spatial tensor field.

Stride determines:

    how far the observation window jumps

Stride = 1

    dense neighborhood traversal

Stride = 2

    sparse neighborhood traversal

Thus stride changes:

    spatial interaction coverage

---

# Sampling Perspective

Convolution can also be viewed as:

    spatial measurement collection

The kernel acts like:

    a local measurement operator

Stride controls:

    how frequently measurements are taken

This is deeply connected to:

- signal processing
- sampling theory
- spatial compression
- information density

---

# Mathematical Intuition

Output size formula:

"""
# Output dimension formula
# output_size =
# ((input_size - kernel_size + 2*padding) / stride) + 1

"""
Larger stride produces:

    smaller output tensors

because fewer spatial observations occur.

---

# Important Insight

Increasing stride introduces:

    information loss

Why?

Because some local neighborhoods
are never observed.

This creates a tradeoff between:

- computational efficiency
- spatial precision

---

# In This Step

We will explore:

1. dense spatial sampling
2. sparse spatial sampling
3. stride geometry
4. output resolution changes
5. spatial information compression
6. sampling-density tradeoffs
"""

#%%

import torch
import torch.nn.functional as F
import torch.nn as nn

torch.manual_seed(42)

#%% [markdown]
"""
# 1. Creating a Spatial Tensor

We create a simple geometric tensor field.

The values increase spatially so we can easily observe:

- neighborhood traversal
- skipped regions
- sampling density changes
"""

#%%

x = torch.tensor([
    [1.,  2.,  3.,  4.,  5.,  6.],
    [7.,  8.,  9., 10., 11., 12.],
    [13., 14., 15., 16., 17., 18.],
    [19., 20., 21., 22., 23., 24.],
    [25., 26., 27., 28., 29., 30.],
    [31., 32., 33., 34., 35., 36.]
])

print("Input Tensor:")
print(x)

print("\nInput Shape:")
print(x.shape)

#%% [markdown]
"""
# 2. Converting to Conv2D Format

PyTorch Conv2D expects:

    [batch, channels, height, width]
"""

#%%

x_conv = x.reshape(1, 1, 6, 6)

print("Conv Tensor Shape:")
print(x_conv.shape)

#%% [markdown]
"""
# 3. Creating a Simple Averaging Kernel

This kernel performs:

    local neighborhood aggregation

Kernel size:

    3 × 3
"""

#%%

kernel = torch.tensor([
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9]
])

print("Kernel:")
print(kernel)

#%% [markdown]
"""
# 4. Dense Spatial Sampling (Stride = 1)

Stride = 1 means:

    every neighboring region is observed

The kernel moves:

    one spatial step at a time

This creates:

    dense interaction coverage
"""

#%%

dense_output = F.conv2d(
    x_conv,
    kernel.reshape(1, 1, 3, 3),
    stride=1
)

print("Dense Sampling Output:")
print(dense_output.squeeze())

print("\nDense Output Shape:")
print(dense_output.squeeze().shape)

#%% [markdown]
"""
# Understanding Dense Sampling

Observe carefully:

The kernel visits almost every
possible local neighborhood.

This preserves:

- spatial continuity
- local detail
- interaction precision

But computational cost increases because:

    more local interactions are computed
"""

#%% [markdown]
"""
# 5. Sparse Spatial Sampling (Stride = 2)

Stride = 2 means:

    skip intermediate regions

The kernel now jumps:

    two spatial steps

This creates:

    coarser spatial observation
"""

#%%

sparse_output = F.conv2d(
    x_conv,
    kernel.reshape(1, 1, 3, 3),
    stride=2
)

print("Sparse Sampling Output:")
print(sparse_output.squeeze())

print("\nSparse Output Shape:")
print(sparse_output.squeeze().shape)

#%% [markdown]
"""
# Understanding Sparse Sampling

Notice:

- output resolution decreases
- fewer spatial measurements occur
- some local neighborhoods are skipped

This reduces:

    computational complexity

But also reduces:

    spatial precision
"""

#%% [markdown]
"""
# 6. Visualizing Spatial Traversal

Stride = 1 traversal:

    dense overlapping observation

Kernel visits:

    almost every local region

Stride = 2 traversal:

    sparse observation grid

Kernel skips intermediate neighborhoods.

This fundamentally changes:

    interaction geometry
"""

#%%

print("Input Spatial Size:")
print(x.shape)

print("\nStride = 1 Output Size:")
print(dense_output.squeeze().shape)

print("\nStride = 2 Output Size:")
print(sparse_output.squeeze().shape)

#%% [markdown]
"""
# 7. Manual Visualization of Sampling Geometry

Let us manually inspect
which regions are observed.

Kernel size:

    3 × 3
"""

#%%

print("Stride = 1 Sampling Positions")
print("--------------------------------")

for i in range(0, 6 - 3 + 1, 1):

    for j in range(0, 6 - 3 + 1, 1):

        print(f"Top-left position: ({i},{j})")

#%%

print("\nStride = 2 Sampling Positions")
print("--------------------------------")

for i in range(0, 6 - 3 + 1, 2):

    for j in range(0, 6 - 3 + 1, 2):

        print(f"Top-left position: ({i},{j})")

#%% [markdown]
"""
# Important Observation

Stride = 2 skips many regions.

Those neighborhoods never contribute
to the output representation.

This demonstrates:

    information compression through sampling
"""

#%% [markdown]
"""
# 8. Stride and Representation Systems

Stride affects:

- spatial resolution
- interaction density
- representation granularity

Small stride:
    fine-grained representations

Large stride:
    coarse-grained representations

Thus stride controls:

    observation scale
"""

#%% [markdown]
"""
# 9. Hierarchical CNN Systems

Modern CNNs often progressively increase:

    effective observation scope

using:

- convolution stacking
- stride
- pooling

This gradually transforms:

    detailed local geometry
        ↓
    compressed semantic structure

This is one of the core mechanisms
behind hierarchical representation learning.
"""

#%% [markdown]
"""
# 10. Stride as Spatial Compression

A very important perspective:

Stride behaves like:

    learned spatial downsampling

Instead of manually resizing an image,
the network learns representations while:

    reducing observation density

This simultaneously performs:

- feature extraction
- spatial compression

which is computationally efficient.
"""

#%% [markdown]
"""
# Final Conceptual Understanding

Stride controls:

    how densely a tensor field is observed

Small stride:
    dense spatial interaction

Large stride:
    sparse spatial interaction

This creates a tradeoff between:

- detail preservation
- computational efficiency

Stride therefore becomes a critical mechanism for:

- representation scaling
- spatial compression
- hierarchical abstraction

---

# Deep Insight

One of the deepest ideas in deep learning:

    Intelligence often depends on
    what information is sampled
    and what information is ignored.

Stride is one of the first mechanisms
that introduces this principle
into convolution systems.
"""