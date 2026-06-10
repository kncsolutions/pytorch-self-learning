# %% [markdown]
"""
# STEP 04 — BOOLEAN MASKING

## CORE INTUITION

Boolean masking controls information flow.

A mask determines:

- what becomes visible
- what becomes hidden
- what participates in computation
- what gets ignored

Masking is one of the most important ideas
in modern tensor systems.

It appears constantly in:

- transformers
- attention systems
- NLP
- segmentation
- anomaly filtering
- reinforcement learning

This step introduces:

- boolean tensors
- conditional tensor selection
- logical masking
- masked computation
- information filtering systems
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

## Boolean Tensor

A boolean tensor contains:

    True
    False

values.


------------------------------------------------------------

## Mask

A mask is a logical tensor controlling
tensor visibility.


------------------------------------------------------------

## Boolean Indexing

Boolean indexing selects tensor elements
where the mask equals:

    True


------------------------------------------------------------

## Conditional Selection

Masks enable conditional tensor computation.


------------------------------------------------------------

## Information Filtering

Masking filters relevant tensor regions
while ignoring others.


------------------------------------------------------------

## Sparse Visibility

Masks allow partial tensor visibility
without modifying original tensor structure.
"""
# %%


# %% [markdown]
"""
# ============================================================
# SECTION 01 — BASIC BOOLEAN MASKING
# ============================================================

A logical condition creates a boolean mask.
"""
# %%


# ============================================================
# BASIC BOOLEAN MASKING
# ============================================================

tensor = torch.tensor(
    [5, 12, 7, 20, 3, 15]
)

print("\n" + "=" * 80)
print("BASIC BOOLEAN MASKING")
print("=" * 80)

print(f"\nOriginal Tensor:\n{tensor}")

mask = tensor > 10

print(f"\nBoolean Mask:\n{mask}")

filtered_tensor = tensor[mask]

print(f"\nFiltered Tensor:\n{filtered_tensor}")


# %% [markdown]
"""
# MASK GEOMETRY

tensor > 10

creates:

    [False, True, False, True, False, True]

Only positions where:

    mask == True

become visible.
"""
# %%


# %% [markdown]
"""
# ============================================================
# SECTION 02 — BOOLEAN INDEXING
# ============================================================

Masks can directly index tensors.
"""
# %%


# ============================================================
# BOOLEAN INDEXING
# ============================================================

tensor = torch.tensor(
    [2, 4, 6, 8, 10]
)

mask = torch.tensor(
    [True, False, True, False, True]
)

print("\n" + "=" * 80)
print("BOOLEAN INDEXING")
print("=" * 80)

print(f"\nTensor:\n{tensor}")

print(f"\nMask:\n{mask}")

result = tensor[mask]

print(f"\nSelected Elements:\n{result}")


# %% [markdown]
"""
# GEOMETRIC INTERPRETATION

The mask behaves like a visibility filter.

Visible positions:

    True

Hidden positions:

    False
"""
# %%


# %% [markdown]
"""
# ============================================================
# SECTION 03 — CONDITIONAL FILTERING
# ============================================================

Logical conditions dynamically create masks.
"""
# %%


# ============================================================
# CONDITIONAL FILTERING
# ============================================================

tensor = torch.randn(10)

print("\n" + "=" * 80)
print("CONDITIONAL FILTERING")
print("=" * 80)

print(f"\nOriginal Tensor:\n{tensor}")

positive_mask = tensor > 0

positive_values = tensor[positive_mask]

print(f"\nPositive Mask:\n{positive_mask}")

print(f"\nPositive Values:\n{positive_values}")


# %% [markdown]
"""
# MATHEMATICAL FORMULATION

Mask:

    M[i] = True
    if
    Tensor[i] > 0

Selection:

    Output = Tensor[M]
"""
# %%


# %% [markdown]
"""
# ============================================================
# SECTION 04 — MASKING MULTI-DIMENSIONAL TENSORS
# ============================================================

Masking also works on higher-dimensional tensors.
"""
# %%


# ============================================================
# MULTI-DIMENSIONAL MASKING
# ============================================================

tensor = torch.tensor(
    [
        [5, 12, 7],
        [20, 3, 15]
    ]
)

print("\n" + "=" * 80)
print("MULTI-DIMENSIONAL MASKING")
print("=" * 80)

print(f"\nOriginal Tensor:\n{tensor}")

mask = tensor > 10

print(f"\nMask:\n{mask}")

filtered_values = tensor[mask]

print(f"\nFiltered Values:\n{filtered_values}")


# %% [markdown]
"""
# GEOMETRIC OBSERVATION

The mask preserves tensor geometry.

Selection extracts values corresponding
to True positions.
"""
# %%


# %% [markdown]
"""
# ============================================================
# SECTION 05 — MASKED ASSIGNMENT
# ============================================================

Masks can modify tensor regions selectively.
"""
# %%


# ============================================================
# MASKED ASSIGNMENT
# ============================================================

tensor = torch.tensor(
    [1, 5, 10, 15, 20]
)

print("\n" + "=" * 80)
print("MASKED ASSIGNMENT")
print("=" * 80)

print(f"\nOriginal Tensor:\n{tensor}")

mask = tensor > 10

tensor[mask] = -1

print(f"\nModified Tensor:\n{tensor}")


# %% [markdown]
"""
# MASKED COMPUTATION

Only positions satisfying the logical condition
were modified.
"""
# %%


# %% [markdown]
"""
# ============================================================
# SECTION 06 — torch.where()
# ============================================================

torch.where() performs conditional selection.
"""
# %%


# ============================================================
# TORCH WHERE
# ============================================================

tensor = torch.tensor(
    [5, 12, 7, 20]
)

print("\n" + "=" * 80)
print("TORCH WHERE")
print("=" * 80)

print(f"\nOriginal Tensor:\n{tensor}")

result = torch.where(
    tensor > 10,
    torch.tensor(1),
    torch.tensor(0)
)

print(f"\nResult:\n{result}")


# %% [markdown]
"""
# MATHEMATICAL INTERPRETATION

For each tensor element:

    if condition == True:
        output = value_a

    else:
        output = value_b
"""
# %%


# %% [markdown]
"""
# ============================================================
# SECTION 07 — COMBINING MASKS
# ============================================================

Logical operations can combine masks.
"""
# %%


# ============================================================
# COMBINING MASKS
# ============================================================

tensor = torch.tensor(
    [5, 12, 18, 25, 30]
)

print("\n" + "=" * 80)
print("COMBINING MASKS")
print("=" * 80)

print(f"\nOriginal Tensor:\n{tensor}")

mask_1 = tensor > 10

mask_2 = tensor < 25

combined_mask = mask_1 & mask_2

print(f"\nMask 1:\n{mask_1}")

print(f"\nMask 2:\n{mask_2}")

print(f"\nCombined Mask:\n{combined_mask}")

result = tensor[combined_mask]

print(f"\nFiltered Result:\n{result}")


# %% [markdown]
"""
# LOGICAL GEOMETRY

AND operation:

    condition_1 AND condition_2

Only elements satisfying both conditions
remain visible.
"""
# %%


# %% [markdown]
"""
# ============================================================
# SECTION 08 — MASKING IMAGE TENSORS
# ============================================================

Masking becomes extremely important in
computer vision systems.
"""
# %%


# ============================================================
# IMAGE-LIKE TENSOR
# ============================================================

image_tensor = torch.randn(
    3, 4, 4
)

print("\n" + "=" * 80)
print("IMAGE TENSOR MASKING")
print("=" * 80)

print(
    f"\nImage Tensor Shape:\n"
    f"{image_tensor.shape}"
)

high_intensity_mask = image_tensor > 0.5

high_values = image_tensor[
    high_intensity_mask
]

print(
    f"\nNumber of High Intensity Values:\n"
    f"{high_values.numel()}"
)


# %% [markdown]
"""
# COMPUTER VISION CONNECTION

Masks can isolate:

- edges
- objects
- regions
- activations
- segmentation zones
"""
# %%


# %% [markdown]
"""
# ============================================================
# SECTION 09 — ATTENTION MASK INTUITION
# ============================================================

Transformers use masks extensively.

Example:

- hide future tokens
- ignore padding
- control attention visibility
"""
# %%


# ============================================================
# SIMPLE ATTENTION MASK
# ============================================================

attention_scores = torch.tensor(
    [
        [1.2, 0.8, 0.5],
        [0.7, 1.5, 0.3],
        [0.2, 0.4, 1.8]
    ]
)

mask = torch.tensor(
    [
        [True, True, False],
        [True, True, False],
        [True, True, False]
    ]
)

print("\n" + "=" * 80)
print("ATTENTION MASK INTUITION")
print("=" * 80)

print(f"\nAttention Scores:\n{attention_scores}")

print(f"\nMask:\n{mask}")

masked_scores = attention_scores.masked_fill(
    ~mask,
    float("-inf")
)

print(f"\nMasked Scores:\n{masked_scores}")


# %% [markdown]
"""
# ATTENTION INTUITION

Masked positions become inaccessible.

The mask controls:

    information visibility

inside the attention system.
"""
# %%


# %% [markdown]
"""
# ============================================================
# SECTION 10 — MEMORY AND MASKING
# ============================================================

Masks themselves are tensors.

Large-scale masking systems may consume
significant memory.
"""
# %%


# ============================================================
# MEMORY INFORMATION
# ============================================================

tensor = torch.randn(1000)

mask = tensor > 0

print("\n" + "=" * 80)
print("MEMORY INFORMATION")
print("=" * 80)

print(f"\nTensor Shape : {tensor.shape}")

print(f"Mask Shape   : {mask.shape}")

print(
    f"\nTensor Memory : "
    f"{tensor.numel() * tensor.element_size()} Bytes"
)

print(
    f"Mask Memory   : "
    f"{mask.numel() * mask.element_size()} Bytes"
)


# %% [markdown]
"""
# MEMORY OBSERVATION

Boolean tensors also occupy memory.

Large attention masks can become computationally
expensive in transformer systems.
"""
# %%


# %% [markdown]
"""
# ============================================================
# IMPORTANT DEEP LEARNING CONNECTIONS
# ============================================================

Boolean masking appears constantly in:

- transformers
- attention systems
- NLP
- segmentation
- anomaly detection
- sparse computation
- reinforcement learning
- token filtering
- recommendation systems

Understanding masking deeply is foundational
for modern AI systems.
"""
# %%


# %% [markdown]
"""
# ============================================================
# FINAL CONCEPTUAL SUMMARY
# ============================================================

Boolean masking controls tensor visibility.

It enables:

- selective computation
- conditional routing
- sparse interaction
- information filtering

Masking is not merely:

    conditional selection

It is:

    structured information flow control
    inside tensor systems.
"""
# %%