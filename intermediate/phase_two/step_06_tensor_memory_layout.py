
# %% [markdown]
"""
# STEP 06 — TENSOR MEMORY LAYOUT

## CORE INTUITION

A tensor is not only geometry.

A tensor is also:

    structured memory

Tensor operations affect:

- memory organization
- computational efficiency
- tensor views
- data locality
- storage behavior

Deep learning performance depends heavily on:

    tensor memory layout

This step introduces:

- contiguous tensors
- tensor storage
- memory views
- reshape vs view
- transpose behavior
- contiguous memory systems
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

## Tensor Storage

Tensor values are stored inside
linear memory.


------------------------------------------------------------

## Contiguous Tensor

A contiguous tensor stores values
in sequential memory order.


------------------------------------------------------------

## Tensor View

A view creates a new perspective
over existing memory.


------------------------------------------------------------

## Memory Layout

Memory layout describes how tensor values
are physically arranged in memory.


------------------------------------------------------------

## Stride

Stride determines how memory positions
are traversed across dimensions.


------------------------------------------------------------

## Data Locality

Sequential memory access improves
computational efficiency.
"""
# %%


# %% [markdown]
"""
# ============================================================
# SECTION 01 — TENSOR STORAGE
# ============================================================

Tensors are stored linearly in memory.
"""
# %%


# ============================================================
# BASIC TENSOR STORAGE
# ============================================================

tensor = torch.tensor(
    [
        [1, 2, 3],
        [4, 5, 6]
    ]
)

print("\n" + "=" * 80)
print("TENSOR STORAGE")
print("=" * 80)

print(f"\nTensor:\n{tensor}")

print(f"\nTensor Shape : {tensor.shape}")

print(f"Tensor Stride: {tensor.stride()}")

print(f"Is Contiguous: {tensor.is_contiguous()}")


# %% [markdown]
"""
# MEMORY INTERPRETATION

Although the tensor appears 2-dimensional:

    [2, 3]

memory itself is linear.

Storage order:

    [1, 2, 3, 4, 5, 6]
"""
# %%


# %% [markdown]
"""
# ============================================================
# SECTION 02 — UNDERSTANDING STRIDE
# ============================================================

Stride controls memory traversal.
"""
# %%


# ============================================================
# STRIDE EXAMPLE
# ============================================================

tensor = torch.tensor(
    [
        [10, 20, 30],
        [40, 50, 60]
    ]
)

print("\n" + "=" * 80)
print("TENSOR STRIDE")
print("=" * 80)

print(f"\nTensor:\n{tensor}")

print(f"\nShape : {tensor.shape}")

print(f"Stride: {tensor.stride()}")


# %% [markdown]
"""
# STRIDE GEOMETRY

Stride:

    (3, 1)

means:

To move across rows:

    jump 3 memory positions

To move across columns:

    jump 1 memory position
"""
# %%


# %% [markdown]
"""
# ============================================================
# SECTION 03 — VIEWS AND SHARED MEMORY
# ============================================================

Many tensor operations create views.

Views share underlying memory.
"""
# %%


# ============================================================
# VIEW EXAMPLE
# ============================================================

original_tensor = torch.tensor(
    [
        [1, 2, 3],
        [4, 5, 6]
    ]
)

view_tensor = original_tensor.view(3, 2)

print("\n" + "=" * 80)
print("TENSOR VIEWS")
print("=" * 80)

print(f"\nOriginal Tensor:\n{original_tensor}")

print(f"\nView Tensor:\n{view_tensor}")

view_tensor[0, 0] = 999

print("\nModified View Tensor.")

print(f"\nOriginal Tensor:\n{original_tensor}")

print(f"\nView Tensor:\n{view_tensor}")


# %% [markdown]
"""
# MEMORY OBSERVATION

The view shares memory with
the original tensor.

Changing one tensor affects the other.
"""
# %%


# %% [markdown]
"""
# ============================================================
# SECTION 04 — RESHAPE VS VIEW
# ============================================================

reshape() and view() appear similar
but behave differently internally.
"""
# %%


# ============================================================
# RESHAPE VS VIEW
# ============================================================

tensor = torch.arange(12)

print("\n" + "=" * 80)
print("RESHAPE VS VIEW")
print("=" * 80)

print(f"\nOriginal Tensor:\n{tensor}")

view_tensor = tensor.view(3, 4)

reshape_tensor = tensor.reshape(2, 6)

print(f"\nView Tensor:\n{view_tensor}")

print(f"\nReshape Tensor:\n{reshape_tensor}")


# %% [markdown]
"""
# IMPORTANT DISTINCTION

view():

    requires contiguous memory

reshape():

    may create new memory if needed

reshape() is generally safer.
"""
# %%


# %% [markdown]
"""
# ============================================================
# SECTION 05 — TRANSPOSE AND MEMORY LAYOUT
# ============================================================

transpose() changes tensor geometry.
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

transposed_tensor = tensor.transpose(0, 1)

print("\n" + "=" * 80)
print("TRANSPOSE AND MEMORY")
print("=" * 80)

print(f"\nOriginal Tensor:\n{tensor}")

print(f"\nTransposed Tensor:\n{transposed_tensor}")

print(f"\nOriginal Stride  : {tensor.stride()}")

print(
    f"Transposed Stride: "
    f"{transposed_tensor.stride()}"
)

print(
    f"\nIs Contiguous:\n"
    f"{transposed_tensor.is_contiguous()}"
)


# %% [markdown]
"""
# GEOMETRIC OBSERVATION

transpose():

    changes dimension ordering

but does not physically rearrange memory.

This often creates:

    non-contiguous tensors
"""
# %%


# %% [markdown]
"""
# ============================================================
# SECTION 06 — CONTIGUOUS MEMORY
# ============================================================

contiguous() reorganizes tensor memory
into sequential layout.
"""
# %%


# ============================================================
# CONTIGUOUS EXAMPLE
# ============================================================

tensor = torch.tensor(
    [
        [1, 2, 3],
        [4, 5, 6]
    ]
)

transposed_tensor = tensor.transpose(0, 1)

contiguous_tensor = transposed_tensor.contiguous()

print("\n" + "=" * 80)
print("CONTIGUOUS MEMORY")
print("=" * 80)

print(
    f"\nTransposed Contiguous:\n"
    f"{transposed_tensor.is_contiguous()}"
)

print(
    f"Contiguous Tensor:\n"
    f"{contiguous_tensor.is_contiguous()}"
)


# %% [markdown]
"""
# MEMORY REORGANIZATION

contiguous():

    physically reorganizes memory

into sequential layout.
"""
# %%


# %% [markdown]
"""
# ============================================================
# SECTION 07 — VIEW FAILURE EXAMPLE
# ============================================================

view() fails on some non-contiguous tensors.
"""
# %%


# ============================================================
# VIEW FAILURE
# ============================================================

tensor = torch.randn(2, 3)

transposed_tensor = tensor.transpose(0, 1)

print("\n" + "=" * 80)
print("VIEW FAILURE")
print("=" * 80)

print(
    f"\nIs Contiguous:\n"
    f"{transposed_tensor.is_contiguous()}"
)

try:

    result = transposed_tensor.view(6)

except Exception as error:

    print("\nView Operation Failed.")

    print(f"\nError:\n{error}")


# %% [markdown]
"""
# FAILURE ANALYSIS

view() expects:

    contiguous memory layout

Non-contiguous tensors may require:

    contiguous()

before reshaping.
"""
# %%


# %% [markdown]
"""
# ============================================================
# SECTION 08 — CONTIGUOUS FIX
# ============================================================

contiguous() resolves memory layout issues.
"""
# %%


# ============================================================
# CONTIGUOUS FIX
# ============================================================

tensor = torch.randn(2, 3)

transposed_tensor = tensor.transpose(0, 1)

fixed_tensor = (
    transposed_tensor
    .contiguous()
    .view(6)
)

print("\n" + "=" * 80)
print("CONTIGUOUS FIX")
print("=" * 80)

print(f"\nFixed Tensor:\n{fixed_tensor}")


# %% [markdown]
"""
# COMPUTATIONAL INTERPRETATION

Tensor geometry and memory layout
are tightly connected.
"""
# %%


# %% [markdown]
"""
# ============================================================
# SECTION 09 — STORAGE SIZE
# ============================================================

Tensors consume memory based on:

- element count
- datatype size
"""
# %%


# ============================================================
# STORAGE INFORMATION
# ============================================================

tensor = torch.randn(100, 100)

print("\n" + "=" * 80)
print("STORAGE INFORMATION")
print("=" * 80)

print(f"\nTensor Shape : {tensor.shape}")

print(f"Tensor Dtype : {tensor.dtype}")

print(f"Tensor Elements : {tensor.numel()}")

print(
    f"Element Size : "
    f"{tensor.element_size()} Bytes"
)

print(
    f"Total Memory : "
    f"{tensor.numel() * tensor.element_size()} Bytes"
)


# %% [markdown]
"""
# MEMORY FORMULATION

Total Memory:

    number_of_elements × bytes_per_element
"""
# %%


# %% [markdown]
"""
# ============================================================
# SECTION 10 — HIGHER DIMENSION MEMORY LAYOUT
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
    8, 3, 224, 224
)

print("\n" + "=" * 80)
print("HIGHER DIMENSION MEMORY")
print("=" * 80)

print(f"\nTensor Shape : {tensor_4d.shape}")

print(f"Tensor Stride: {tensor_4d.stride()}")

print(
    f"\nIs Contiguous:\n"
    f"{tensor_4d.is_contiguous()}"
)


# %% [markdown]
"""
# DEEP LEARNING CONNECTION

Memory layout heavily affects:

- GPU efficiency
- convolution speed
- tensor throughput
- cache locality
- training performance
"""
# %%


# %% [markdown]
"""
# ============================================================
# IMPORTANT DEEP LEARNING CONNECTIONS
# ============================================================

Tensor memory layout appears constantly in:

- CNNs
- transformers
- CUDA kernels
- tensor optimization
- GPU acceleration
- scientific computing
- distributed systems

Understanding memory layout deeply is foundational
for high-performance tensor systems.
"""
# %%


# %% [markdown]
"""
# ============================================================
# FINAL CONCEPTUAL SUMMARY
# ============================================================

A tensor is simultaneously:

- geometry
- memory
- computation

Tensor operations affect:

- structural layout
- memory traversal
- computational efficiency

Memory layout is not merely:

    implementation detail

It is:

    a foundational component of
    tensor system performance.
"""
# %%

