# %% [markdown]
"""
# STEP 03 — TENSOR INDEXING

## CORE INTUITION

Indexing is tensor navigation.

Tensor indexing allows us to:

- access information
- select regions
- isolate structures
- navigate dimensions
- manipulate tensor geometry

Deep learning systems constantly perform indexing operations.

Examples:

- selecting batches
- extracting channels
- masking tokens
- attention routing
- sequence slicing

This step introduces:

- tensor indexing
- slicing
- advanced indexing
- boolean indexing
- tensor navigation systems
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

## Tensor Indexing

Tensor indexing refers to selecting tensor elements
through positional navigation.


------------------------------------------------------------

## Tensor Slicing

Slicing extracts continuous tensor regions.

Example:

    tensor[0:3]


------------------------------------------------------------

## Dimension Navigation

Each index corresponds to movement across
tensor dimensions.


------------------------------------------------------------

## Boolean Indexing

Boolean indexing selects tensor elements
using logical masks.


------------------------------------------------------------

## Advanced Indexing

Advanced indexing allows selecting arbitrary
tensor locations.


------------------------------------------------------------

## Tensor Geometry

Tensor indexing changes visible tensor structure
without necessarily changing underlying memory.
"""
# %%


# %% [markdown]
"""
# ============================================================
# SECTION 01 — BASIC TENSOR INDEXING
# ============================================================

Tensor:

    [Rows, Columns]

Indexing allows selecting individual elements.
"""
# %%


# ============================================================
# BASIC INDEXING
# ============================================================

tensor = torch.tensor(
    [
        [10, 20, 30],
        [40, 50, 60],
        [70, 80, 90]
    ]
)

print("\n" + "=" * 80)
print("BASIC TENSOR INDEXING")
print("=" * 80)

print(f"\ntensor:\n{tensor}")

print(f"\ntensor shape : {tensor.shape}")

print(f"\nFirst Element : {tensor[0, 0]}")

print(f"Center Element: {tensor[1, 1]}")

print(f"Last Element  : {tensor[2, 2]}")


# %% [markdown]
"""
# INDEX GEOMETRY

tensor[row, column]

Example:

    tensor[1, 2]

means:

    row index    -> 1
    column index -> 2
"""
# %%


# %% [markdown]
"""
# ============================================================
# SECTION 02 — ROW AND COLUMN SELECTION
# ============================================================

Entire tensor regions can be extracted.
"""
# %%


# ============================================================
# ROW AND COLUMN INDEXING
# ============================================================

print("\n" + "=" * 80)
print("ROW AND COLUMN SELECTION")
print("=" * 80)

first_row = tensor[0]

second_column = tensor[:, 1]

print(f"\nFirst Row:\n{first_row}")

print(f"\nSecond Column:\n{second_column}")


# %% [markdown]
"""
# GEOMETRIC INTERPRETATION

tensor[0]

selects:

    row 0

tensor[:, 1]

means:

    all rows
    column 1
"""
# %%


# %% [markdown]
"""
# ============================================================
# SECTION 03 — TENSOR SLICING
# ============================================================

Slicing extracts continuous tensor regions.

Syntax:

    tensor[start:end]
"""
# %%


# ============================================================
# TENSOR SLICING
# ============================================================

print("\n" + "=" * 80)
print("TENSOR SLICING")
print("=" * 80)

sub_tensor = tensor[0:2, 1:3]

print(f"\nOriginal Tensor:\n{tensor}")

print(f"\nSliced Tensor:\n{sub_tensor}")

print(f"\nSliced Shape : {sub_tensor.shape}")


# %% [markdown]
"""
# SLICE GEOMETRY

tensor[0:2, 1:3]

means:

    rows    -> 0 to 1
    columns -> 1 to 2
"""
# %%


# %% [markdown]
"""
# ============================================================
# SECTION 04 — NEGATIVE INDEXING
# ============================================================

Negative indexing navigates from the end
of tensor dimensions.
"""
# %%


# ============================================================
# NEGATIVE INDEXING
# ============================================================

print("\n" + "=" * 80)
print("NEGATIVE INDEXING")
print("=" * 80)

print(f"\nLast Row:\n{tensor[-1]}")

print(f"\nLast Column:\n{tensor[:, -1]}")

print(f"\nBottom Right Element:\n{tensor[-1, -1]}")


# %% [markdown]
"""
# NEGATIVE INDEX GEOMETRY

-1 refers to:

    final position

-2 refers to:

    second last position
"""
# %%


# %% [markdown]
"""
# ============================================================
# SECTION 05 — HIGHER DIMENSION INDEXING
# ============================================================

Deep learning tensors often contain
multiple semantic dimensions.

Example:

    [Batch, Channels, Height, Width]
"""
# %%


# ============================================================
# HIGHER DIMENSION TENSOR
# ============================================================

tensor_4d = torch.randn(
    2, 3, 4, 4
)

print("\n" + "=" * 80)
print("HIGHER DIMENSION INDEXING")
print("=" * 80)

print(f"\ntensor shape : {tensor_4d.shape}")


# %% [markdown]
"""
# DIMENSION INTERPRETATION

Shape:

    [2, 3, 4, 4]

means:

    2 -> batch size
    3 -> channels
    4 -> height
    4 -> width
"""
# %%


# ============================================================
# HIGHER DIMENSION NAVIGATION
# ============================================================

first_image = tensor_4d[0]

red_channel = tensor_4d[0, 0]

print(f"\nFirst Image Shape : {first_image.shape}")

print(f"Red Channel Shape : {red_channel.shape}")


# %% [markdown]
"""
# GEOMETRIC NAVIGATION

tensor_4d[0]

selects:

    first batch sample

tensor_4d[0, 0]

selects:

    first channel
    of first image
"""
# %%


# %% [markdown]
"""
# ============================================================
# SECTION 06 — BOOLEAN INDEXING
# ============================================================

Boolean indexing selects tensor elements
using logical conditions.
"""
# %%


# ============================================================
# BOOLEAN INDEXING
# ============================================================

tensor = torch.tensor(
    [5, 12, 7, 20, 3, 15]
)

print("\n" + "=" * 80)
print("BOOLEAN INDEXING")
print("=" * 80)

print(f"\nOriginal Tensor:\n{tensor}")

mask = tensor > 10

print(f"\nBoolean Mask:\n{mask}")

filtered_tensor = tensor[mask]

print(f"\nFiltered Tensor:\n{filtered_tensor}")


# %% [markdown]
"""
# BOOLEAN GEOMETRY

tensor > 10

creates:

    logical selection structure

Only positions where:

    mask == True

are extracted.
"""
# %%


# %% [markdown]
"""
# ============================================================
# SECTION 07 — ADVANCED INDEXING
# ============================================================

Advanced indexing selects arbitrary tensor positions.
"""
# %%


# ============================================================
# ADVANCED INDEXING
# ============================================================

tensor = torch.tensor(
    [
        [10, 20, 30],
        [40, 50, 60],
        [70, 80, 90]
    ]
)

row_indices = torch.tensor([0, 2])

column_indices = torch.tensor([1, 2])

print("\n" + "=" * 80)
print("ADVANCED INDEXING")
print("=" * 80)

print(f"\nOriginal Tensor:\n{tensor}")

selected_elements = tensor[
    row_indices,
    column_indices
]

print(f"\nSelected Elements:\n{selected_elements}")


# %% [markdown]
"""
# INDEX PAIR INTERPRETATION

Selections:

    tensor[0, 1]
    tensor[2, 2]

Results:

    20
    90
"""
# %%


# %% [markdown]
"""
# ============================================================
# SECTION 08 — ELLIPSIS INDEXING
# ============================================================

Ellipsis (...) simplifies indexing
across higher dimensions.
"""
# %%


# ============================================================
# ELLIPSIS INDEXING
# ============================================================

tensor_4d = torch.randn(
    2, 3, 4, 4
)

print("\n" + "=" * 80)
print("ELLIPSIS INDEXING")
print("=" * 80)

print(f"\nTensor Shape : {tensor_4d.shape}")

last_dimension = tensor_4d[..., 0]

print(
    f"\nLast Dimension Slice Shape : "
    f"{last_dimension.shape}"
)


# %% [markdown]
"""
# ELLIPSIS INTERPRETATION

tensor[..., 0]

means:

    preserve all previous dimensions
    select index 0 from final dimension
"""
# %%


# %% [markdown]
"""
# ============================================================
# SECTION 09 — INDEXING AND MEMORY
# ============================================================

Many indexing operations create views
instead of copying memory.
"""
# %%


# ============================================================
# MEMORY VIEW EXAMPLE
# ============================================================

tensor = torch.tensor(
    [
        [1, 2, 3],
        [4, 5, 6]
    ]
)

slice_tensor = tensor[:, :2]

print("\n" + "=" * 80)
print("INDEXING AND MEMORY")
print("=" * 80)

print(f"\nOriginal Tensor:\n{tensor}")

print(f"\nSlice Tensor:\n{slice_tensor}")

slice_tensor[0, 0] = 999

print("\nModified Slice Tensor.")

print(f"\nOriginal Tensor:\n{tensor}")

print(f"\nSlice Tensor:\n{slice_tensor}")


# %% [markdown]
"""
# MEMORY OBSERVATION

The slice operation created a view.

Changing the slice also modified
the original tensor memory.
"""
# %%


# %% [markdown]
"""
# ============================================================
# IMPORTANT DEEP LEARNING CONNECTIONS
# ============================================================

Tensor indexing appears constantly in:

- attention masking
- sequence modeling
- token selection
- channel extraction
- image cropping
- batch sampling
- reinforcement learning
- sparse tensor systems

Understanding indexing deeply is foundational
for advanced tensor systems.
"""
# %%


# %% [markdown]
"""
# ============================================================
# FINAL CONCEPTUAL SUMMARY
# ============================================================

Tensor indexing is structured tensor navigation.

It allows:

- dimensional traversal
- region extraction
- selective computation
- information routing

Indexing is not merely:

    element access

It is:

    navigation through tensor geometry.
"""
# %%