#%% [markdown]
"""
# Step 07 — Feature Maps

# Core Philosophy

Feature maps are not merely:

    "detected features"

Feature maps are:

    interaction response fields

A convolution kernel defines:

    a local interaction rule

As the kernel moves through the tensor field,
different regions respond differently.

The resulting tensor becomes:

    a spatial response map

showing how strongly each region reacts to
a particular interaction rule.

---

# Important Conceptual Shift

Most beginner explanations say:

    "CNN detects edges"

But a deeper interpretation is:

    convolution generates structured
    interaction response geometry

A feature map represents:

    response intensity distribution

across spatial regions.

---

# Geometric Interpretation

Input tensor:
    spatial interaction surface

Kernel:
    local interaction operator

Feature map:
    response field generated from interaction

Thus the process becomes:

    local structure
        ↓
    interaction
        ↓
    response intensity
        ↓
    representation field

---

# Important Insight

Different kernels generate different:

- interaction sensitivities
- spatial response patterns
- representation structures

Thus different feature maps capture different:

    aspects of tensor geometry

---

# Multi-Channel Perspective

Modern CNNs do not generate:

    one feature map

They generate:

    many parallel response fields

Each feature map learns different:

- spatial sensitivities
- interaction rules
- representation patterns

This creates:

    distributed representation systems

---

# Mathematical Intuition

For a convolution operation:

"""
# Convolution formulation
# y(i,j) = \sum_m \sum_n x(i+m,j+n) w(m,n)

"""
The feature map stores:

    interaction responses

for all spatial locations.

---

# In This Step

We will explore:

1. feature maps as response fields
2. different kernel sensitivities
3. multi-feature extraction
4. parallel representation systems
5. spatial response geometry
6. learned interaction behavior
"""

#%%

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

#%% [markdown]
"""
# 1. Creating a Spatial Tensor

We create a tensor with:

- smooth regions
- strong transitions
- structured spatial variation

This helps visualize:

    different interaction responses
"""

#%%

x = torch.tensor([
    [1., 1., 1., 1., 1., 1.],
    [1., 1., 1., 1., 1., 1.],
    [1., 1., 10., 10., 10., 10.],
    [1., 1., 10., 10., 10., 10.],
    [1., 1., 10., 10., 10., 10.],
    [1., 1., 10., 10., 10., 10.]
])

print("Input Tensor:")
print(x)

print("\nInput Shape:")
print(x.shape)

#%% [markdown]
"""
# 2. Converting to Conv2D Format

PyTorch expects:

    [batch, channels, height, width]
"""

#%%

x_conv = x.reshape(1, 1, 6, 6)

print("Conv Tensor Shape:")
print(x_conv.shape)

#%% [markdown]
"""
# 3. Vertical Interaction Kernel

This kernel becomes sensitive to:

    left-right spatial variation

Conceptually:

    directional interaction sensitivity
"""

#%%

vertical_kernel = torch.tensor([
    [1., 0., -1.],
    [1., 0., -1.],
    [1., 0., -1.]
])

print("Vertical Kernel:")
print(vertical_kernel)

#%% [markdown]
"""
# 4. Horizontal Interaction Kernel

This kernel becomes sensitive to:

    top-bottom spatial variation

Again:

    interaction sensitivity differs
"""

#%%

horizontal_kernel = torch.tensor([
    [1., 1., 1.],
    [0., 0., 0.],
    [-1., -1., -1.]
])

print("Horizontal Kernel:")
print(horizontal_kernel)

#%% [markdown]
"""
# 5. Averaging Kernel

This kernel performs:

    local smoothing interaction

It responds to:

    regional consistency
"""

#%%

average_kernel = torch.tensor([
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9]
])

print("Average Kernel:")
print(average_kernel)

#%% [markdown]
"""
# 6. Generating Feature Maps

Each kernel creates:

    a different response field

The same input tensor produces:

    different representations

depending on interaction rules.
"""

#%%

vertical_feature_map = F.conv2d(
    x_conv,
    vertical_kernel.reshape(1, 1, 3, 3)
)

horizontal_feature_map = F.conv2d(
    x_conv,
    horizontal_kernel.reshape(1, 1, 3, 3)
)

average_feature_map = F.conv2d(
    x_conv,
    average_kernel.reshape(1, 1, 3, 3)
)

#%% [markdown]
"""
# 7. Vertical Response Field

This feature map shows:

    where horizontal spatial transitions occur

High responses indicate:

    strong directional contrast
"""

#%%

print("Vertical Feature Map:")
print(vertical_feature_map.squeeze())

#%% [markdown]
"""
# 8. Horizontal Response Field

This feature map responds differently.

It emphasizes:

    vertical structural changes
"""

#%%

print("Horizontal Feature Map:")
print(horizontal_feature_map.squeeze())

#%% [markdown]
"""
# 9. Averaging Response Field

This feature map performs:

    regional aggregation

The representation becomes smoother because:

    neighboring values cooperate
"""

#%%

print("Average Feature Map:")
print(average_feature_map.squeeze())

#%% [markdown]
"""
# Important Observation

Same input tensor.
Different kernels.
Different feature maps.

Why?

Because each kernel defines:

    different interaction behavior

Thus feature maps are fundamentally:

    response distributions
"""

#%% [markdown]
"""
# 10. Multi-Feature Representation Systems

Modern CNNs use:

    many kernels simultaneously

Each kernel learns different:

- local interaction rules
- spatial sensitivities
- representation preferences

Thus networks build:

    distributed representation fields
"""

#%%

multi_kernel = torch.stack([
    vertical_kernel,
    horizontal_kernel,
    average_kernel
])

multi_kernel = multi_kernel.unsqueeze(1)

print("Multi-Kernel Shape:")
print(multi_kernel.shape)

#%% [markdown]
"""
# 11. Parallel Feature Map Generation

Now multiple kernels operate simultaneously.

The output contains:

    multiple response fields

This is one of the foundational mechanisms
behind convolutional representation learning.
"""

#%%

multi_feature_maps = F.conv2d(
    x_conv,
    multi_kernel
)

print("Multi-Feature Map Shape:")
print(multi_feature_maps.shape)

#%% [markdown]
"""
# Output Shape Interpretation

Output shape:

    [batch, channels, height, width]

Important insight:

Each output channel becomes:

    a separate feature map

Thus output channels represent:

    parallel representation systems
"""

#%%

print("Feature Map 1 — Vertical Response")
print(multi_feature_maps[0, 0])

print("\nFeature Map 2 — Horizontal Response")
print(multi_feature_maps[0, 1])

print("\nFeature Map 3 — Averaging Response")
print(multi_feature_maps[0, 2])

#%% [markdown]
"""
# 12. Feature Maps as Representation Fields

A feature map should not be viewed as:

    symbolic understanding

Instead it represents:

    spatial interaction response intensity

The network gradually learns:

    which local interactions matter

through training and optimization.
"""

#%% [markdown]
"""
# 13. Hierarchical Feature Systems

Shallow feature maps often respond to:

- edges
- textures
- local contrast

Deeper feature maps respond to:

- structures
- arrangements
- semantic compositions

This hierarchy emerges through:

    repeated interaction aggregation
"""

#%% [markdown]
"""
# 14. Feature Maps and Deep Learning

One of the most important ideas in CNNs:

Different kernels create different:

    representation perspectives

The network therefore develops:

    distributed spatial understanding

rather than one single representation.
"""

#%% [markdown]
"""
# Final Conceptual Understanding

Feature maps are fundamentally:

    interaction response fields

A kernel defines:

    what local interactions matter

The feature map records:

    where those interactions occur strongly

Modern deep learning systems build intelligence through:

    many interacting response fields

operating simultaneously across tensor geometry.

---

# Deep Insight

One of the deepest principles in CNN systems:

    Representation emerges from
    structured interaction responses.

Feature maps are the spatial memory
of those interactions.
"""