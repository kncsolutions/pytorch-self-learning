# %% [markdown]
"""
# STEP 02 — BROADCASTING GEOMETRY

## CORE INTUITION

Broadcasting is not simply automatic expansion.

Broadcasting is:

    geometric tensor alignment

PyTorch attempts to align tensor structures before
performing computation.

This step explores:

- tensor geometry
- dimensional alignment
- singleton expansion
- geometric compatibility
- higher-dimensional broadcasting

The goal is not merely to run operations.

The goal is to understand:

    how tensor structures negotiate interaction.
"""
# %%


# ============================================================
# IMPORTS
# ============================================================

import torch


# %% [markdown]
"""
# ============================================================
# GLOSSARY
# ============================================================

## Tensor Geometry

Tensor geometry refers to the structural arrangement
of tensor dimensions.

Example:

    [Batch, Channels, Height, Width]

is not merely:

    four numbers

It encodes:

- tensor populations
- information fields
- spatial structure


------------------------------------------------------------

## Broadcasting

Broadcasting is PyTorch's mechanism for aligning
tensor geometries before computation.


------------------------------------------------------------

## Singleton Dimension

A singleton dimension has size:

    1

Example:

    [1, 4]

Singleton dimensions can expand during broadcasting.


------------------------------------------------------------

## Shape Compatibility

Two dimensions are compatible if:

1. They are equal

OR

2. One of them equals:

    1


------------------------------------------------------------

## Implicit Expansion

Broadcasting creates virtual expansion
without physically copying tensor memory.


------------------------------------------------------------

## Tensor Alignment

Tensor operations require compatible geometry.

PyTorch aligns dimensions from:

    right to left


------------------------------------------------------------

## Element-wise Operations

Element-wise operations apply independently
to corresponding tensor elements.

Example:

    A[i, j] + B[i, j]
"""
# %%


# %% [markdown]
"""
# ============================================================
# SECTION 01 — BASIC GEOMETRIC ALIGNMENT
# ============================================================

Tensor A:

    [3, 4]

Tensor B:

    [1, 4]

Alignment:

    [3, 4]
    [1, 4]
        ↓
    [3, 4]

The singleton dimension expands.
"""
# %%


# ============================================================
# BASIC BROADCASTING
# ============================================================

tensor_a = torch.tensor(
    [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12]
    ]
)

tensor_b = torch.tensor(
    [
        [100, 200, 300, 400]
    ]
)

print("\n" + "=" * 80)
print("BASIC GEOMETRIC ALIGNMENT")
print("=" * 80)

print(f"tensor_a shape : {tensor_a.shape}")

print(f"tensor_b shape : {tensor_b.shape}")

print(f"\ntensor_a:\n{tensor_a}")

print(f"\ntensor_b:\n{tensor_b}")

result = tensor_a + tensor_b

print(f"\nresult:\n{result}")


# %% [markdown]
"""
# OBSERVATION

tensor_b does not physically become:

    [3, 4]

PyTorch performs:

    implicit geometric expansion

without copying memory.
"""
# %%


# %% [markdown]
"""
# ============================================================
# SECTION 02 — RIGHT TO LEFT ALIGNMENT
# ============================================================

PyTorch compares tensor dimensions:

    right to left

Example:

Tensor A:

    [2, 3, 4]

Tensor B:

        [3, 4]

PyTorch internally interprets:

    [2, 3, 4]
    [1, 3, 4]

The leading singleton dimension broadcasts.
"""
# %%


# ============================================================
# RIGHT TO LEFT ALIGNMENT
# ============================================================

tensor_a = torch.randn(2, 3, 4)

tensor_b = torch.randn(3, 4)

print("\n" + "=" * 80)
print("RIGHT TO LEFT ALIGNMENT")
print("=" * 80)

print(f"tensor_a shape : {tensor_a.shape}")

print(f"tensor_b shape : {tensor_b.shape}")

result = tensor_a + tensor_b

print(f"\nresult shape : {result.shape}")


# %% [markdown]
"""
# MATHEMATICAL FORMULATION

Tensor alignment occurs dimension-wise.

Example:

    [2, 3, 4]
    [1, 3, 4]

Compatibility Check:

    4 ↔ 4   ✓
    3 ↔ 3   ✓
    2 ↔ 1   ✓

Result:

    [2, 3, 4]
"""
# %%


# %% [markdown]
"""
# ============================================================
# SECTION 03 — COLUMN-WISE GEOMETRY
# ============================================================

Broadcasting can occur vertically.

Tensor A:

    [2, 3]

Tensor B:

    [2, 1]

Alignment:

    [2, 3]
    [2, 1]
        ↓
    [2, 3]

The singleton dimension expands across columns.
"""
# %%


# ============================================================
# COLUMN-WISE BROADCASTING
# ============================================================

tensor_a = torch.tensor(
    [
        [1, 2, 3],
        [4, 5, 6]
    ]
)

tensor_b = torch.tensor(
    [
        [10],
        [20]
    ]
)

print("\n" + "=" * 80)
print("COLUMN-WISE BROADCASTING")
print("=" * 80)

print(f"tensor_a shape : {tensor_a.shape}")

print(f"tensor_b shape : {tensor_b.shape}")

result = tensor_a + tensor_b

print(f"\nresult:\n{result}")


# %% [markdown]
"""
# GEOMETRIC INTERPRETATION

tensor_b behaves like:

    [
        [10, 10, 10],
        [20, 20, 20]
    ]

But PyTorch avoids physically copying memory.
"""
# %%


# %% [markdown]
"""
# ============================================================
# SECTION 04 — UNSQUEEZE AND GEOMETRY CREATION
# ============================================================

unsqueeze() creates singleton dimensions.

This is extremely important because:

    singleton dimensions enable broadcasting.
"""
# %%


# ============================================================
# UNSQUEEZE EXAMPLE
# ============================================================

vector = torch.tensor([1, 2, 3])

print("\n" + "=" * 80)
print("UNSQUEEZE GEOMETRY")
print("=" * 80)

print(f"original shape : {vector.shape}")

row_vector = vector.unsqueeze(0)

column_vector = vector.unsqueeze(1)

print(f"\nrow vector shape    : {row_vector.shape}")

print(f"column vector shape : {column_vector.shape}")

print(f"\nrow vector:\n{row_vector}")

print(f"\ncolumn vector:\n{column_vector}")


# %% [markdown]
"""
# GEOMETRIC OBSERVATION

unsqueeze(0):

    [3]
        ↓
    [1, 3]

unsqueeze(1):

    [3]
        ↓
    [3, 1]

Dimension placement changes broadcasting behavior.
"""
# %%


# %% [markdown]
"""
# ============================================================
# SECTION 05 — HIGHER DIMENSION BROADCASTING
# ============================================================

Broadcasting becomes extremely important in deep learning.

Example tensor:

    [Batch, Channels, Height, Width]

Normalization tensor:

    [1, Channels, 1, 1]

This allows channel-wise normalization across
entire tensor populations.
"""
# %%


# ============================================================
# HIGHER DIMENSION BROADCASTING
# ============================================================

batch_tensor = torch.randn(
    8, 3, 224, 224
)

channel_bias = torch.randn(
    1, 3, 1, 1
)

print("\n" + "=" * 80)
print("HIGHER DIMENSION BROADCASTING")
print("=" * 80)

print(
    f"batch tensor shape : "
    f"{batch_tensor.shape}"
)

print(
    f"channel bias shape : "
    f"{channel_bias.shape}"
)

result = batch_tensor + channel_bias

print(f"\nresult shape : {result.shape}")


# %% [markdown]
"""
# MATHEMATICAL ALIGNMENT

Tensor A:

    [8, 3, 224, 224]

Tensor B:

    [1, 3, 1, 1]

Compatibility:

    224 ↔ 1    ✓
    224 ↔ 1    ✓
    3   ↔ 3    ✓
    8   ↔ 1    ✓

Result:

    [8, 3, 224, 224]
"""
# %%


# %% [markdown]
"""
# ============================================================
# SECTION 06 — BROADCASTING FAILURE
# ============================================================

Broadcasting fails when dimensions are incompatible.
"""
# %%


# ============================================================
# BROADCASTING FAILURE
# ============================================================

tensor_a = torch.randn(2, 3)

tensor_b = torch.randn(4, 3)

print("\n" + "=" * 80)
print("BROADCASTING FAILURE")
print("=" * 80)

print(f"tensor_a shape : {tensor_a.shape}")

print(f"tensor_b shape : {tensor_b.shape}")

try:

    result = tensor_a + tensor_b

except Exception as error:

    print("\nBroadcasting Failed.")

    print(f"\nError:\n{error}")


# %% [markdown]
"""
# FAILURE ANALYSIS

Alignment:

    [2, 3]
    [4, 3]

Compatibility Check:

    3 ↔ 3   ✓
    2 ↔ 4   ✗

Since neither dimension equals:

    1

broadcasting becomes impossible.
"""
# %%


# %% [markdown]
"""
# ============================================================
# SECTION 07 — EXPAND VS REPEAT
# ============================================================

expand():

    virtual expansion

repeat():

    physical duplication
"""
# %%


# ============================================================
# EXPAND VS REPEAT
# ============================================================

tensor = torch.tensor([[1, 2, 3]])

expanded_tensor = tensor.expand(4, 3)

repeated_tensor = tensor.repeat(4, 1)

print("\n" + "=" * 80)
print("EXPAND VS REPEAT")
print("=" * 80)

print(f"original shape : {tensor.shape}")

print(f"expanded shape : {expanded_tensor.shape}")

print(f"repeated shape : {repeated_tensor.shape}")

print(
    f"\noriginal storage size : "
    f"{tensor.storage().size()}"
)

print(
    f"expanded storage size : "
    f"{expanded_tensor.storage().size()}"
)

print(
    f"repeated storage size : "
    f"{repeated_tensor.storage().size()}"
)


# %% [markdown]
"""
# MEMORY OBSERVATION

expand():

    reuses memory

repeat():

    allocates additional memory

This distinction becomes important in large-scale
deep learning systems.
"""
# %%


# %% [markdown]
"""
# ============================================================
# FINAL CONCEPTUAL SUMMARY
# ============================================================

Broadcasting is one of the foundational ideas
in tensor systems.

It enables:

- efficient tensor interaction
- geometric compatibility
- scalable deep learning computation
- implicit dimensional alignment

Broadcasting is not merely:

    automatic expansion

It is:

    structured tensor geometry negotiation.
"""
# %%