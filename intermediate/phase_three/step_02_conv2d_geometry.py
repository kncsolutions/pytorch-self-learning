#%% [markdown]
"""
# Step 02 — Conv2D Geometry

# Core Philosophy


Conv2D is:

    structured spatial tensor interaction

A Conv2D system allows tensors to interact
through localized geometric neighborhoods.

Instead of processing an entire image globally,
Conv2D performs:

    local spatial aggregation

This creates:

- spatial interaction systems
- geometric response fields
- hierarchical representations

---

# From Conv1D to Conv2D

Conv1D interacted with:

    sequential neighborhoods

Conv2D interacts with:

    spatial neighborhoods

You can think of it like, Conv1D operates on a sublist
whereas Conv2D operates on a sub matrix.

Input tensor becomes:

    height × width

or more generally:

    channels × height × width

The tensor is now treated as:

    a geometric interaction surface

---

# Core Mathematical Formulation

For a 2D convolution:

y(i,j) = Σ Σ x(i+m,j+n) * w(m,n)

More formally:

"""
# Mathematical representation
# y(i,j) = \sum_m \sum_n x(i+m,j+n) w(m,n)

"""
Where:

- x = input tensor
- w = kernel tensor
- y = output tensor
- (i,j) = spatial position

---

# Geometric Interpretation

A kernel slides spatially across the tensor field.

At each location:

    local region
        ↓
    weighted interaction
        ↓
    aggregated response

The output tensor becomes:

    an interaction response field

---

# Important Insight

Conv2D does NOT understand:

- objects
- cats
- edges
- meaning

It only computes:

    local numerical interactions

Higher-level representations emerge later through:

- repeated convolution
- stacking
- hierarchy formation

---

# In This Step

We will explore:

1. 2D tensor geometry
2. Spatial neighborhoods
3. Manual Conv2D intuition
4. Kernel interaction systems
5. PyTorch Conv2D
6. Feature map generation
"""

#%%

import torch
import torch.nn as nn

torch.manual_seed(42)

#%% [markdown]
"""
# 1. Creating a Simple 2D Tensor

This tensor can represent:

- grayscale image
- spatial measurements
- heat map
- geometric signal surface

We intentionally use small dimensions
to clearly visualize local interactions.
"""

#%%

x = torch.tensor([
    [1., 2., 3., 4.],
    [5., 6., 7., 8.],
    [9., 10., 11., 12.],
    [13., 14., 15., 16.]
])

print("Input Tensor:")
print(x)

print("\nTensor Shape:")
print(x.shape)

#%% [markdown]
"""
# 2. Understanding Spatial Geometry

The tensor is now treated as:

    a spatial interaction surface

Each value has:

- horizontal relationships
- vertical relationships
- local neighborhood structure

This is fundamentally different from:

    flat sequential interaction
"""

#%% [markdown]
"""
# 3. Creating a Kernel

Kernel:

    [[ 1,  0],
     [ 0, -1]]

This kernel becomes sensitive to:

    local contrast differences

But conceptually it represents:

    a local interaction rule
"""

#%%

kernel = torch.tensor([
    [1., 0.],
    [0., -1.]
])

print("Kernel:")
print(kernel)

print("\nKernel Shape:")
print(kernel.shape)

#%% [markdown]
"""
# 4. Manual Conv2D Intuition

Kernel size:

    2 × 2

Sliding interaction windows:

Window 1:
    [[1,2],
     [5,6]]

Window 2:
    [[2,3],
     [6,7]]

and so on...

At each location:

    local spatial region
        ↓
    weighted interaction
        ↓
    aggregated response
"""

#%%

output = []

kernel_height = kernel.shape[0]
kernel_width = kernel.shape[1]

input_height = x.shape[0]
input_width = x.shape[1]

for i in range(input_height - kernel_height + 1):

    row_outputs = []

    for j in range(input_width - kernel_width + 1):

        # Extract local spatial neighborhood
        window = x[
            i:i + kernel_height,
            j:j + kernel_width
        ]

        # Local interaction
        interaction = window * kernel

        # Aggregation
        response = interaction.sum()

        print(f"\nPosition ({i},{j})")
        print("-------------------------")

        print("Window:")
        print(window)

        print("\nInteraction:")
        print(interaction)

        print("\nAggregated Response:")
        print(response)

        row_outputs.append(response)

    output.append(torch.tensor(row_outputs))

output = torch.stack(output)

#%% [markdown]
"""
# 5. Final Conv2D Output

The output tensor represents:

    spatial interaction responses

Each value tells us:

    how strongly a local region
    responds to the kernel rule
"""

#%%

print("\nFinal Output Tensor:")
print(output)

print("\nOutput Shape:")
print(output.shape)

#%% [markdown]
"""
# 6. Understanding Feature Maps

The output tensor is often called:

    a feature map

But conceptually it is better understood as:

    an interaction response field

Each output position represents:

    local response intensity

to a particular interaction rule.
"""

#%% [markdown]
"""
# 7. Conv2D in PyTorch

PyTorch Conv2D expects shape:

    [batch, channels, height, width]

Why?

Because Conv2D operates on:

    geometric tensor fields
"""

#%%

x_conv = x.reshape(1, 1, 4, 4)

print("Conv2D Input Shape:")
print(x_conv.shape)

#%% [markdown]
"""
# 8. Creating Conv2D Layer

We create:

- 1 input channel
- 1 output channel
- 2×2 kernel

The layer learns:

    local spatial interaction rules
"""

#%%

conv2d = nn.Conv2d(
    in_channels=1,
    out_channels=1,
    kernel_size=2,
    bias=False
)

#%% [markdown]
"""
# 9. Manually Assigning Kernel Weights

We manually assign:

    [[1,0],
     [0,-1]]

This helps compare:

- manual interaction computation
- PyTorch Conv2D computation
"""

#%%

with torch.no_grad():

    conv2d.weight[:] = torch.tensor([
        [
            [
                [1., 0.],
                [0., -1.]
            ]
        ]
    ])

print("Conv2D Weights:")
print(conv2d.weight)

#%% [markdown]
"""
# 10. Running Conv2D

PyTorch now performs:

    localized spatial aggregation

automatically across the tensor field.
"""

#%%

y = conv2d(x_conv)

print("Raw Conv2D Output:")
print(y)

#%% [markdown]
"""
# 11. Simplifying Dimensions

Current output shape:

    [batch, channels, height, width]

We remove unnecessary dimensions
for easier visualization.
"""

#%%

y_squeezed = y.squeeze()

print("Simplified Output:")
print(y_squeezed)

#%% [markdown]
"""
# 12. Comparing Manual vs PyTorch Outputs
"""

#%%

print("Manual Conv2D Output:")
print(output)

print("\nPyTorch Conv2D Output:")
print(y_squeezed)

#%% [markdown]
"""
# 13. Spatial Interaction Geometry

Observe carefully:

The kernel never sees:

    the entire tensor globally

It only sees:

    local geometric neighborhoods

Yet after many layers:

- global structures emerge
- semantic representations emerge
- hierarchical understanding emerges

This is one of the most profound ideas
in deep learning systems.

---

# Local Interactions → Global Representations

Deep networks repeatedly apply:

    local interaction rules

Layer after layer.

Eventually:

- textures emerge
- shapes emerge
- object structures emerge
- semantic representations emerge

This hierarchy originates from:

    localized tensor field interactions
"""

#%% [markdown]
"""
# Final Conceptual Understanding

Conv2D systems transform tensors into:

    structured geometric interaction fields

Kernels become:

    local interaction operators

Feature maps become:

    response surfaces

Deep learning systems progressively construct:

    hierarchical spatial representations

through repeated local aggregation.

---

# Deep Insight

One of the deepest principles in AI:

    Complex intelligence can emerge
    from repeated local interactions.

Conv2D systems are one of the clearest
demonstrations of this principle.
"""