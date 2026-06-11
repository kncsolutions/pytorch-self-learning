#%% [markdown]
"""
# Step 09 — Pooling as Information Compression

# Core Philosophy

Pooling is not merely:

    "reducing tensor size"

Pooling performs:

    localized information compression

A convolution system often generates:

- large feature maps
- dense spatial responses
- redundant local information

Pooling compresses these response fields while attempting to preserve:

    important interaction signals

---

# Important Intuition

Pooling answers an important question:

    "What information should survive
    after local aggregation?"

Instead of preserving every spatial value,
pooling summarizes local neighborhoods.

This creates:

- compressed representations
- reduced computation
- larger effective receptive fields
- abstraction growth

---

# Geometric Interpretation

A pooling window slides through
a feature map.

At each local region:

    neighborhood responses
        ↓
    local summarization
        ↓
    compressed representation

Thus pooling behaves like:

    structured spatial compression

---

# Important Insight

Pooling intentionally introduces:

    information loss

This is NOT necessarily bad.

The goal is:

    preserve important structure
    while reducing redundancy

This creates a tradeoff between:

- detail preservation
- representation compactness

---

# Max Pooling vs Average Pooling

Max pooling:
    preserve strongest local activation

Average pooling:
    preserve local statistical summary

They represent different:

    compression philosophies

---

# Mathematical Intuition

Max Pooling:

"""
# Max pooling
# y(i,j) = max(local_region)

"""

Average Pooling:

"""
# Average pooling
# y(i,j) = average(local_region)

"""

Pooling transforms:

    dense response geometry
        ↓
    compressed interaction summaries

---

# Important Conceptual Shift

Pooling should not be viewed as:

    "shrinking tensors"

Instead think:

    selective information preservation

The network decides:

    what local information matters most

---

# In This Step

We will explore:

1. max pooling
2. average pooling
3. local information compression
4. spatial summarization
5. representation compactness
6. information preservation tradeoffs
"""

#%%

import torch
import torch.nn.functional as F

torch.manual_seed(42)

#%% [markdown]
"""
# 1. Creating a Feature Map

We create a synthetic feature map
with regions of:

- weak activation
- strong activation
- local structure

This helps visualize:

    pooling behavior
"""

#%%

x = torch.tensor([
    [1., 1., 2., 2., 1., 1.],
    [1., 5., 6., 2., 1., 1.],
    [2., 6., 9., 3., 2., 1.],
    [1., 2., 3., 8., 7., 2.],
    [1., 1., 2., 7., 9., 3.],
    [1., 1., 1., 2., 3., 1.]
])

print("Input Feature Map:")
print(x)

print("\nInput Shape:")
print(x.shape)

#%% [markdown]
"""
# 2. Converting to Pooling Format

PyTorch pooling expects:

    [batch, channels, height, width]
"""

#%%

x_pool = x.reshape(1, 1, 6, 6)

print("Pooling Tensor Shape:")
print(x_pool.shape)

#%% [markdown]
"""
# 3. Max Pooling Intuition

Max pooling preserves:

    strongest local response

Pooling window:

    2 × 2

Stride:

    2

This creates non-overlapping
compression regions.
"""

#%%

max_pooled = F.max_pool2d(
    x_pool,
    kernel_size=2,
    stride=2
)

print("Max Pooled Output:")
print(max_pooled.squeeze())

print("\nMax Pool Shape:")
print(max_pooled.squeeze().shape)

#%% [markdown]
"""
# Understanding Max Pooling

Each 2×2 neighborhood becomes:

    one representative value

The strongest activation survives.

This assumes:

    dominant activations carry
    important local information

Thus max pooling performs:

    competitive local selection
"""

#%% [markdown]
"""
# 4. Manual Max Pooling Visualization

Let us inspect local compression regions.
"""

#%%

for i in range(0, 6, 2):

    for j in range(0, 6, 2):

        region = x[i:i+2, j:j+2]

        print(f"\nRegion ({i}:{i+2}, {j}:{j+2})")
        print(region)

        print("Max Response:")
        print(region.max())

#%% [markdown]
"""
# Important Observation

Many values disappear.

Only dominant responses survive.

Thus pooling introduces:

    controlled information reduction
"""

#%% [markdown]
"""
# 5. Average Pooling Intuition

Average pooling computes:

    local statistical summaries

Instead of selecting the strongest response,
it preserves:

    regional average behavior
"""

#%%

avg_pooled = F.avg_pool2d(
    x_pool,
    kernel_size=2,
    stride=2
)

print("Average Pooled Output:")
print(avg_pooled.squeeze())

print("\nAverage Pool Shape:")
print(avg_pooled.squeeze().shape)

#%% [markdown]
"""
# Understanding Average Pooling

Average pooling preserves:

    regional statistical structure

This creates smoother representations.

Unlike max pooling:

    all values contribute

Thus average pooling performs:

    cooperative local summarization
"""

#%% [markdown]
"""
# 6. Comparing Compression Philosophies

Max pooling:
    dominant activation preservation

Average pooling:
    regional statistical preservation

Different pooling methods preserve different:

- interaction signals
- representation properties
- information priorities
"""

#%%

print("Max Pooling Result:")
print(max_pooled.squeeze())

print("\nAverage Pooling Result:")
print(avg_pooled.squeeze())

#%% [markdown]
"""
# 7. Pooling and Spatial Compression

Observe carefully:

Input shape:
    6 × 6

Pooled shape:
    3 × 3

Pooling reduces:

- spatial resolution
- computational cost
- representation size

while attempting to preserve:

    meaningful interaction structure
"""

#%%

print("Original Shape:")
print(x.shape)

print("\nPooled Shape:")
print(max_pooled.squeeze().shape)

#%% [markdown]
"""
# 8. Pooling and Receptive Fields

Pooling indirectly increases:

    effective receptive field size

Why?

Because compressed representations now summarize:

    larger spatial regions

Thus deeper layers gradually observe:

    broader contextual structure
"""

#%% [markdown]
"""
# 9. Pooling as Abstraction

Pooling encourages networks to focus less on:

    exact spatial location

and more on:

    dominant structural presence

This contributes to:

- translation robustness
- representation abstraction
- hierarchical understanding
"""

#%% [markdown]
"""
# 10. Important Tradeoff

Pooling creates a major tradeoff.

Benefits:

- reduced computation
- larger context
- compressed representation

Costs:

- spatial precision loss
- information removal
- reduced localization accuracy

Thus pooling is always a balance between:

    efficiency and detail
"""

#%% [markdown]
"""
# 11. Modern Perspective on Pooling

Some modern architectures reduce pooling usage.

Why?

Because aggressive pooling may destroy:

- fine spatial detail
- localization precision
- small structural information

Especially in:

- segmentation systems
- medical imaging
- dense prediction tasks

careful information preservation becomes important.
"""

#%% [markdown]
"""
# 12. Pooling and Biological Intuition

Pooling resembles a simplified form of:

    selective attention

The system asks:

    "What local information is most important?"

Compression therefore becomes:

    importance-driven summarization
"""

#%% [markdown]
"""
# Final Conceptual Understanding

Pooling is fundamentally:

    localized information compression

It transforms:

    dense interaction response fields

into:

    compact representation summaries

Pooling preserves selected local structure
while reducing:

- redundancy
- spatial resolution
- computational complexity

---

# Deep Insight

One of the deepest principles in deep learning:

    Intelligence often depends on
    what information is preserved
    and what information is discarded.

Pooling is one of the first major mechanisms
that operationalizes this principle.
"""