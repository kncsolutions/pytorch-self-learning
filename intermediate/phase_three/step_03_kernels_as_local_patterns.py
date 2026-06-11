#%% [markdown]
"""
# Step 03 — Kernels as Local Patterns

# Core Philosophy

Kernels are not magical feature detectors.

Kernels are:

    localized interaction operators

A kernel defines:

    how neighboring tensor values interact

Different kernels create different kinds of:

- interaction sensitivity
- local aggregation behavior
- spatial response patterns

---

# Important Conceptual Shift

Most tutorials explain kernels as:

    edge detectors
    blur filters
    sharpen filters

But a deeper interpretation is:

    kernels encode local interaction rules

The observed patterns emerge from:

    tensor neighborhood interactions

---

# Core Mathematical Formulation

For Conv2D:

y(i,j) = Σ Σ x(i+m,j+n) * w(m,n)

"""
# Mathematical representation
# y(i,j) = \sum_m \sum_n x(i+m,j+n) w(m,n)

"""
Where:

- x = input tensor
- w = kernel
- y = output response field

The kernel defines:

    interaction weighting geometry

---

# Geometric Interpretation

A kernel slides through local regions.

At each location:

    local tensor neighborhood
        ↓
    weighted interaction
        ↓
    response generation

Different kernels emphasize different:

- spatial relationships
- directional structures
- interaction behaviors

---

# In This Step

We will explore:

1. kernels as interaction systems
2. edge-sensitive kernels
3. averaging kernels
4. sharpening kernels
5. directional interaction behavior
6. feature response geometry
"""

#%%

import torch
import torch.nn.functional as F

torch.manual_seed(42)

#%% [markdown]
"""
# 1. Creating a Spatial Tensor

This tensor represents:

    a geometric interaction surface

We intentionally create:

- low values on left
- high values on right

This creates:

    strong spatial transitions
"""

#%%

x = torch.tensor([
    [1., 1., 1., 1., 1.],
    [1., 1., 1., 1., 1.],
    [1., 1., 10., 10., 10.],
    [1., 1., 10., 10., 10.],
    [1., 1., 10., 10., 10.]
])

print("Input Tensor:")
print(x)

#%% [markdown]
"""
# 2. Converting to Conv2D Format

PyTorch Conv2D expects:

    [batch, channels, height, width]
"""

#%%

x_conv = x.reshape(1, 1, 5, 5)

print("Conv Tensor Shape:")
print(x_conv.shape)

#%% [markdown]
"""
# 3. Kernel 1 — Vertical Difference Sensitivity

Kernel:

    [[ 1,  0, -1],
     [ 1,  0, -1],
     [ 1,  0, -1]]

This kernel becomes sensitive to:

    horizontal spatial changes

Conceptually:

    compare left vs right neighborhoods

This is NOT magic edge detection.

It is:

    directional interaction sensitivity
"""

#%%

vertical_kernel = torch.tensor([
    [1., 0., -1.],
    [1., 0., -1.],
    [1., 0., -1.]
])

print("Vertical Interaction Kernel:")
print(vertical_kernel)

#%% [markdown]
"""
# 4. Applying Vertical Interaction Kernel

The kernel scans spatial neighborhoods.

Each output value becomes:

    local directional interaction response
"""

#%%

vertical_output = F.conv2d(
    x_conv,
    vertical_kernel.reshape(1, 1, 3, 3)
)

print("Vertical Interaction Response:")
print(vertical_output.squeeze())

#%% [markdown]
"""
# Understanding the Output

High magnitude values indicate:

    strong directional differences

The kernel strongly responds where:

    local left-right structure changes rapidly

This demonstrates:

    kernels encode interaction sensitivity
"""

#%% [markdown]
"""
# 5. Kernel 2 — Averaging Kernel

Kernel:

    every position contributes equally

This creates:

    local smoothing behavior

Conceptually:

    neighborhood averaging interaction
"""

#%%

average_kernel = torch.tensor([
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9]
])

print("Averaging Kernel:")
print(average_kernel)

#%% [markdown]
"""
# 6. Applying Averaging Kernel

The kernel aggregates:

    local regional statistics

This reduces:

- sharp variation
- local discontinuities
- spatial contrast
"""

#%%

average_output = F.conv2d(
    x_conv,
    average_kernel.reshape(1, 1, 3, 3)
)

print("Averaging Response:")
print(average_output.squeeze())

#%% [markdown]
"""
# Understanding Averaging Behavior

The output becomes smoother because:

    neighboring values cooperate

The kernel promotes:

    local consistency

This demonstrates:

    kernels shape interaction dynamics
"""

#%% [markdown]
"""
# 7. Kernel 3 — Sharpening Kernel

Kernel:

    [[ 0, -1,  0],
     [-1,  5, -1],
     [ 0, -1,  0]]

Conceptually:

    amplify local contrast structure

The center becomes dominant while
neighboring regions are suppressed.
"""

#%%

sharpen_kernel = torch.tensor([
    [0., -1., 0.],
    [-1., 5., -1.],
    [0., -1., 0.]
])

print("Sharpen Kernel:")
print(sharpen_kernel)

#%% [markdown]
"""
# 8. Applying Sharpening Kernel

This interaction rule increases:

    local structural emphasis
"""

#%%

sharpen_output = F.conv2d(
    x_conv,
    sharpen_kernel.reshape(1, 1, 3, 3)
)

print("Sharpened Response:")
print(sharpen_output.squeeze())

#%% [markdown]
"""
# Understanding Sharpening Behavior

The kernel strengthens:

    local contrast dominance

Important idea:

Different kernels produce different:

- interaction geometries
- response behaviors
- representation sensitivities
"""

#%% [markdown]
"""
# 9. Comparing Multiple Kernels

Observe carefully:

Same input tensor.
Different kernels.
Different response fields.

Why?

Because kernels define:

    interaction behavior

The input remains identical.

Only the local interaction rule changes.
"""

#%%

print("Input Tensor:")
print(x)

print("\nVertical Interaction Response:")
print(vertical_output.squeeze())

print("\nAveraging Response:")
print(average_output.squeeze())

print("\nSharpened Response:")
print(sharpen_output.squeeze())

#%% [markdown]
"""
# 10. Kernels as Learned Interaction Systems

In deep learning:

    kernels are NOT manually designed

Neural networks learn kernels automatically.

During training:

- useful interaction rules emerge
- response sensitivity evolves
- representation systems form

The network gradually learns:

    which local interactions matter

This is a foundational principle behind:

- CNNs
- vision systems
- audio systems
- scientific tensor models

---

# Local Pattern Formation

A kernel can become sensitive to:

- orientation
- texture
- local contrast
- frequency patterns
- spatial continuity

But these are emergent interaction behaviors,
not symbolic understanding.
"""

#%% [markdown]
"""
# 11. Interaction Geometry Perspective

A convolution kernel behaves like:

    a localized interaction operator

It defines:

- what neighboring relationships matter
- how aggregation occurs
- which structures become amplified

Feature maps therefore become:

    interaction response surfaces

This is a much deeper interpretation than:

    "the network detects edges"

---

# Deep Insight

One of the most important ideas in deep learning:

    Representation emerges from interaction sensitivity.

Kernels define:

    what local structures become visible
"""