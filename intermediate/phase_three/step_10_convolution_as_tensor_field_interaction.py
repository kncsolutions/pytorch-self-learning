#%% [markdown]
"""
# Step 10 — Convolution as Tensor Field Interaction

# Core Philosophy

Convolution is not merely:

    "applying filters"

Convolution is:

    structured tensor field interaction

A tensor field contains:

- local structure
- spatial relationships
- interaction neighborhoods
- geometric continuity

Convolution defines how these local regions:

    interact, aggregate, and propagate

through the representation system.

---

# Important Conceptual Shift

Throughout earlier modules we explored:

- kernels
- receptive fields
- stride
- padding
- feature maps
- pooling

Now we unify them into one larger idea:

    convolution systems are local interaction fields

This is one of the foundational ideas behind:

- CNNs
- vision systems
- diffusion systems
- scientific simulations
- neural operators
- PDE-inspired AI systems

---

# Tensor Field Perspective

Instead of thinking:

    image = collection of pixels

we now think:

    tensor = structured interaction field

Each spatial region influences:

    neighboring regions

through learned interaction rules.

This transforms tensors into:

    dynamic representation systems

---

# Geometric Interpretation

A convolution kernel behaves like:

    a local interaction operator

At every spatial location:

    local neighborhood
        ↓
    weighted interaction
        ↓
    aggregation
        ↓
    response propagation

This process repeats hierarchically.

Eventually:

    complex representation geometry emerges

---

# Important Insight

Convolution systems reveal something profound:

    global structure can emerge
    from repeated local interactions

This idea appears throughout science:

- physics
- cellular systems
- graph systems
- biological systems
- fluid simulation
- neural systems

Deep learning strongly relies on this principle.

---

# Mathematical Formulation

For Conv2D:

"""
# Convolution formulation
# y(i,j) = \sum_m \sum_n x(i+m,j+n) w(m,n)

"""

But conceptually this means:

    local regions exchange influence
    through interaction weights

The kernel defines:

    how information propagates locally

---

# Information Flow Perspective

Convolution performs:

    structured information routing

Feature maps become:

    propagated interaction responses

Stacked layers gradually create:

- broader contextual awareness
- hierarchical structure
- semantic representation geometry

---

# Important Connection

Earlier we learned:

- broadcasting = tensor alignment
- masking = selective visibility
- gather/scatter = information routing

Convolution now extends these ideas into:

    local spatial interaction systems

Thus modern AI increasingly becomes:

    structured tensor interaction engineering

---

# In This Step

We will explore:

1. tensor fields
2. local interaction propagation
3. stacked interaction systems
4. feature evolution
5. hierarchical information flow
6. representation emergence
"""

#%%

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(42)

#%% [markdown]
"""
# 1. Creating a Spatial Tensor Field

We create a structured spatial tensor.

The tensor contains:

- local structure
- central activation
- geometric organization

This helps visualize:

    interaction propagation
"""

#%%

x = torch.tensor([
    [0., 0., 1., 1., 1., 0., 0.],
    [0., 1., 2., 2., 2., 1., 0.],
    [1., 2., 5., 5., 5., 2., 1.],
    [1., 2., 5., 10., 5., 2., 1.],
    [1., 2., 5., 5., 5., 2., 1.],
    [0., 1., 2., 2., 2., 1., 0.],
    [0., 0., 1., 1., 1., 0., 0.]
])

print("Input Tensor Field:")
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

x_conv = x.reshape(1, 1, 7, 7)

print("Tensor Field Shape:")
print(x_conv.shape)

#%% [markdown]
"""
# 3. Defining Local Interaction Operators

We create multiple kernels.

Each kernel defines different:

- interaction sensitivities
- propagation behavior
- response dynamics
"""

#%%

vertical_kernel = torch.tensor([
    [1., 0., -1.],
    [1., 0., -1.],
    [1., 0., -1.]
])

horizontal_kernel = torch.tensor([
    [1., 1., 1.],
    [0., 0., 0.],
    [-1., -1., -1.]
])

average_kernel = torch.tensor([
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9]
])

#%% [markdown]
"""
# 4. Creating Parallel Interaction Systems

Each kernel becomes:

    a separate local interaction operator

Together they form:

    multiple tensor interaction channels
"""

#%%

kernels = torch.stack([
    vertical_kernel,
    horizontal_kernel,
    average_kernel
])

kernels = kernels.unsqueeze(1)

print("Kernel Tensor Shape:")
print(kernels.shape)

#%% [markdown]
"""
# Shape Interpretation

Shape:

    [3, 1, 3, 3]

Meaning:

- 3 output interaction systems
- 1 input channel
- 3×3 local interaction regions
"""

#%% [markdown]
"""
# 5. Applying Tensor Field Interactions

Now the tensor field undergoes:

    structured local interaction propagation

Each kernel generates:

    a separate response field
"""

#%%

interaction_fields = F.conv2d(
    x_conv,
    kernels,
    padding=1
)

print("Interaction Field Shape:")
print(interaction_fields.shape)

#%% [markdown]
"""
# Output Interpretation

Output shape:

    [batch, channels, height, width]

Each output channel represents:

    a different interaction response field

This creates:

    distributed representation geometry
"""

#%%

print("Vertical Interaction Field:")
print(interaction_fields[0, 0])

print("\nHorizontal Interaction Field:")
print(interaction_fields[0, 1])

print("\nAveraged Interaction Field:")
print(interaction_fields[0, 2])

#%% [markdown]
"""
# 6. Applying Non-Linearity

Non-linearity introduces:

    selective signal propagation

Weak or negative responses become suppressed.

Important idea:

The network gradually learns:

    which interactions remain important
"""

#%%

activated_fields = F.relu(interaction_fields)

print("Activated Interaction Fields Shape:")
print(activated_fields.shape)

#%% [markdown]
"""
# 7. Pooling Interaction Responses

Pooling compresses:

    local interaction information

This creates:

    summarized representation fields
"""

#%%

pooled_fields = F.max_pool2d(
    activated_fields,
    kernel_size=2,
    stride=2
)

print("Pooled Interaction Fields Shape:")
print(pooled_fields.shape)

#%% [markdown]
"""
# Understanding the Full Pipeline

Input Tensor Field
    ↓
Local Interaction Operators
    ↓
Interaction Response Fields
    ↓
Non-Linear Propagation
    ↓
Compressed Representations

This is the foundation of convolutional systems.
"""

#%%

print("Pooled Vertical Field:")
print(pooled_fields[0, 0])

print("\nPooled Horizontal Field:")
print(pooled_fields[0, 1])

print("\nPooled Averaged Field:")
print(pooled_fields[0, 2])

#%% [markdown]
"""
# 8. Stacking Interaction Layers

Modern CNNs repeatedly apply:

- convolution
- activation
- pooling

Each layer transforms:

    previous representation geometry

into:

    more abstract interaction structure

This gradually creates:

    hierarchical representation systems
"""

#%%

conv_layer_2 = nn.Conv2d(
    in_channels=3,
    out_channels=6,
    kernel_size=3,
    padding=1
)

#%% [markdown]
"""
# 9. Running a Second Interaction Layer

Now Layer 2 no longer sees:

    raw tensor geometry

Instead it sees:

    interaction response fields

Thus deeper representation structure emerges.
"""

#%%

layer2_output = conv_layer_2(pooled_fields)

layer2_activated = F.relu(layer2_output)

print("Layer 2 Output Shape:")
print(layer2_activated.shape)

#%% [markdown]
"""
# Important Observation

Representation evolution:

Layer 1:
    local interaction responses

Layer 2:
    interaction-of-interaction representations

This hierarchy allows networks to gradually build:

- structure awareness
- contextual understanding
- semantic organization
"""

#%% [markdown]
"""
# 10. Tensor Field Propagation Perspective

A convolution network behaves like:

    an interaction propagation system

Each layer redistributes:

- local influence
- response intensity
- representation structure

through learned interaction rules.

This resembles many physical systems where:

    local interactions create global behavior
"""

#%% [markdown]
"""
# 11. Connection to Scientific Systems

Convolution systems strongly resemble:

- cellular interaction systems
- field propagation systems
- PDE approximations
- diffusion systems
- physical simulation operators

because all involve:

    local neighborhood interactions

propagating through structured spaces.
"""

#%% [markdown]
"""
# 12. Deep Learning as Tensor Field Engineering

A modern interpretation of CNNs:

Deep learning engineers:

    interaction geometry

The network learns:

- what interactions matter
- how information propagates
- which structures become amplified

Thus CNNs become:

    learned tensor field systems
"""

#%% [markdown]
"""
# Final Conceptual Understanding

Convolution is fundamentally:

    structured tensor field interaction

Kernels define:

    local interaction laws

Feature maps become:

    propagated response fields

Deep networks gradually construct:

    hierarchical representation geometry

through repeated local aggregation,
propagation, compression, and transformation.

---

# Deep Insight

One of the deepest principles in AI:

    Complex global intelligence can emerge
    from repeated local interaction rules.

Convolution systems are one of the clearest
demonstrations of this principle.
"""