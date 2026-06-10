# %% [markdown]
"""
# STEP 07 — TENSOR PERMUTATION SYSTEMS

## CORE INTUITION

Tensor dimensions carry structural meaning.

Changing dimension order changes:

- tensor interpretation
- computational semantics
- interaction structure
- model compatibility

Permutation systems reorganize:

    tensor geometry

without necessarily changing tensor values.

This step introduces:

- permute()
- transpose()
- dimension reordering
- tensor semantic structure
- geometric reinterpretation

These ideas become foundational in:

- computer vision
- transformers
- NLP
- sequence systems
- multimodal systems
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

## Tensor Permutation

Permutation rearranges tensor dimensions.


------------------------------------------------------------

## Dimension Ordering

Tensor dimensions carry semantic meaning.

Example:

    [Batch, Channels, Height, Width]

Changing order changes interpretation.


------------------------------------------------------------

## transpose()

transpose() swaps two dimensions.


------------------------------------------------------------

## permute()

permute() rearranges arbitrary dimension order.


------------------------------------------------------------

## Tensor Geometry

Tensor geometry defines how information
is structurally organized.


------------------------------------------------------------

## Semantic Structure

Tensor dimensions are not merely sizes.

They represent:

- batches
- channels
- sequences
- spatial axes
- feature systems
"""
# %%


# %% [markdown]
"""
# ============================================================
# SECTION 01 — WHY DIMENSION ORDER MATTERS
# ============================================================

Tensor dimensions encode meaning.

Example image tensor:

    [Channels, Height, Width]

Changing order changes interpretation.
"""
# %%


# ============================================================
# IMAGE-LIKE TENSOR
# ============================================================

image_tensor = torch.randn(
    3, 224, 224
)

print("\n" + "=" * 80)
print("DIMENSION ORDER")
print("=" * 80)

print(
    f"\nImage Tensor Shape:\n"
    f"{image_tensor.shape}"
)


# %% [markdown]
"""
# DIMENSION INTERPRETATION

Shape:

    [3, 224, 224]

means:

    3   -> channels
    224 -> height
    224 -> width

Changing dimension order changes tensor meaning.
"""
# %%


# %% [markdown]
"""
# ============================================================
# SECTION 02 — TRANSPOSE
# ============================================================

transpose() swaps two dimensions.
"""
# %%


# ============================================================
# TRANSPOSE EXAMPLE
# ============================================================

tensor = torch.tensor(
    [
        [1, 2, 3],
        [4, 5, 6]
    ]
)

print("\n" + "=" * 80)
print("TRANSPOSE")
print("=" * 80)

print(f"\nOriginal Tensor:\n{tensor}")

print(f"\nOriginal Shape:\n{tensor.shape}")

transposed_tensor = tensor.transpose(0, 1)

print(f"\nTransposed Tensor:\n{transposed_tensor}")

print(
    f"\nTransposed Shape:\n"
    f"{transposed_tensor.shape}"
)


# %% [markdown]
"""
# GEOMETRIC INTERPRETATION

transpose(0, 1)

swaps:

    rows ↔ columns

Original:

    [2, 3]

Result:

    [3, 2]
"""
# %%


# %% [markdown]
"""
# ============================================================
# SECTION 03 — PERMUTE
# ============================================================

permute() allows arbitrary dimension reordering.
"""
# %%


# ============================================================
# PERMUTE EXAMPLE
# ============================================================

tensor = torch.randn(
    2, 3, 4
)

print("\n" + "=" * 80)
print("PERMUTE")
print("=" * 80)

print(f"\nOriginal Shape:\n{tensor.shape}")

permuted_tensor = tensor.permute(
    2, 0, 1
)

print(
    f"\nPermuted Shape:\n"
    f"{permuted_tensor.shape}"
)


# %% [markdown]
"""
# PERMUTATION GEOMETRY

Original:

    [2, 3, 4]

permute(2, 0, 1):

    dimension 2 → position 0
    dimension 0 → position 1
    dimension 1 → position 2

Result:

    [4, 2, 3]
"""
# %%


# %% [markdown]
"""
# ============================================================
# SECTION 04 — IMAGE TENSOR PERMUTATION
# ============================================================

Different frameworks use different
image tensor conventions.
"""
# %%


# ============================================================
# IMAGE PERMUTATION
# ============================================================

image_tensor = torch.randn(
    3, 224, 224
)

print("\n" + "=" * 80)
print("IMAGE TENSOR PERMUTATION")
print("=" * 80)

print(
    f"\nPyTorch Format Shape:\n"
    f"{image_tensor.shape}"
)

image_hwc = image_tensor.permute(
    1, 2, 0
)

print(
    f"\nHWC Format Shape:\n"
    f"{image_hwc.shape}"
)


# %% [markdown]
"""
# FORMAT INTERPRETATION

PyTorch:

    [Channels, Height, Width]

Visualization libraries often expect:

    [Height, Width, Channels]

Permutation converts between representations.
"""
# %%


# %% [markdown]
"""
# ============================================================
# SECTION 05 — SEQUENCE SYSTEMS
# ============================================================

Transformers frequently permute dimensions.
"""
# %%


# ============================================================
# SEQUENCE TENSOR
# ============================================================

sequence_tensor = torch.randn(
    32, 128, 512
)

print("\n" + "=" * 80)
print("SEQUENCE SYSTEMS")
print("=" * 80)

print(
    f"\nOriginal Shape:\n"
    f"{sequence_tensor.shape}"
)

permuted_tensor = sequence_tensor.permute(
    1, 0, 2
)

print(
    f"\nPermuted Shape:\n"
    f"{permuted_tensor.shape}"
)


# %% [markdown]
"""
# TRANSFORMER INTERPRETATION

Original:

    [Batch, Sequence, Features]

Permuted:

    [Sequence, Batch, Features]

Different architectures expect different
dimension conventions.
"""
# %%


# %% [markdown]
"""
# ============================================================
# SECTION 06 — PERMUTATION AND MEMORY
# ============================================================

Permutation changes tensor geometry
without necessarily reorganizing memory.
"""
# %%


# ============================================================
# MEMORY OBSERVATION
# ============================================================

tensor = torch.randn(
    2, 3, 4
)

permuted_tensor = tensor.permute(
    2, 0, 1
)

print("\n" + "=" * 80)
print("PERMUTATION AND MEMORY")
print("=" * 80)

print(
    f"\nOriginal Stride:\n"
    f"{tensor.stride()}"
)

print(
    f"\nPermuted Stride:\n"
    f"{permuted_tensor.stride()}"
)

print(
    f"\nIs Contiguous:\n"
    f"{permuted_tensor.is_contiguous()}"
)


# %% [markdown]
"""
# MEMORY INTERPRETATION

permute():

    changes tensor perspective

without physically rearranging memory.

This often creates:

    non-contiguous tensors
"""
# %%


# %% [markdown]
"""
# ============================================================
# SECTION 07 — CONTIGUOUS AFTER PERMUTATION
# ============================================================

Some operations require contiguous memory.
"""
# %%


# ============================================================
# CONTIGUOUS EXAMPLE
# ============================================================

tensor = torch.randn(
    2, 3, 4
)

permuted_tensor = tensor.permute(
    2, 0, 1
)

contiguous_tensor = (
    permuted_tensor.contiguous()
)

print("\n" + "=" * 80)
print("CONTIGUOUS AFTER PERMUTATION")
print("=" * 80)

print(
    f"\nPermuted Contiguous:\n"
    f"{permuted_tensor.is_contiguous()}"
)

print(
    f"\nContiguous Tensor:\n"
    f"{contiguous_tensor.is_contiguous()}"
)


# %% [markdown]
"""
# GEOMETRIC OBSERVATION

contiguous():

    reorganizes memory physically

to match current tensor geometry.
"""
# %%


# %% [markdown]
"""
# ============================================================
# SECTION 08 — HIGHER DIMENSION PERMUTATION
# ============================================================

Complex tensor systems may contain many dimensions.
"""
# %%


# ============================================================
# HIGHER DIMENSION TENSOR
# ============================================================

tensor_5d = torch.randn(
    8, 16, 32, 64, 128
)

print("\n" + "=" * 80)
print("HIGHER DIMENSION PERMUTATION")
print("=" * 80)

print(
    f"\nOriginal Shape:\n"
    f"{tensor_5d.shape}"
)

permuted_tensor = tensor_5d.permute(
    0, 2, 1, 4, 3
)

print(
    f"\nPermuted Shape:\n"
    f"{permuted_tensor.shape}"
)


# %% [markdown]
"""
# HIGHER DIMENSION INTERPRETATION

Permutation systems become increasingly important
as tensor complexity grows.
"""
# %%


# %% [markdown]
"""
# ============================================================
# SECTION 09 — PERMUTATION FAILURE
# ============================================================

permute() requires valid dimension ordering.
"""
# %%


# ============================================================
# INVALID PERMUTATION
# ============================================================

tensor = torch.randn(
    2, 3, 4
)

print("\n" + "=" * 80)
print("INVALID PERMUTATION")
print("=" * 80)

try:

    invalid_tensor = tensor.permute(
        0, 0, 1
    )

except Exception as error:

    print("\nPermutation Failed.")

    print(f"\nError:\n{error}")


# %% [markdown]
"""
# FAILURE ANALYSIS

Each dimension index must appear:

    exactly once

inside permute().
"""
# %%


# %% [markdown]
"""
# ============================================================
# SECTION 10 — ATTENTION SYSTEM INTUITION
# ============================================================

Attention systems constantly reorganize
tensor dimensions.
"""
# %%


# ============================================================
# ATTENTION-LIKE TENSOR
# ============================================================

attention_tensor = torch.randn(
    8, 12, 128, 64
)

print("\n" + "=" * 80)
print("ATTENTION SYSTEM INTUITION")
print("=" * 80)

print(
    f"\nOriginal Shape:\n"
    f"{attention_tensor.shape}"
)

permuted_attention = attention_tensor.permute(
    0, 2, 1, 3
)

print(
    f"\nPermuted Shape:\n"
    f"{permuted_attention.shape}"
)


# %% [markdown]
"""
# ATTENTION INTERPRETATION

Original:

    [Batch, Heads, Sequence, Features]

Permuted:

    [Batch, Sequence, Heads, Features]

Dimension order changes interaction semantics.
"""
# %%


# %% [markdown]
"""
# ============================================================
# IMPORTANT DEEP LEARNING CONNECTIONS
# ============================================================

Tensor permutation appears constantly in:

- transformers
- attention systems
- NLP
- CNNs
- multimodal systems
- video systems
- scientific computing
- distributed tensor systems

Understanding permutation deeply is foundational
for advanced tensor architectures.
"""
# %%


# %% [markdown]
"""
# ============================================================
# FINAL CONCEPTUAL SUMMARY
# ============================================================

Tensor dimensions encode semantic structure.

Permutation systems reorganize:

- tensor interpretation
- interaction geometry
- computational semantics

Permutation is not merely:

    dimension swapping

It is:

    structured tensor semantic reorganization.
"""
# %%
