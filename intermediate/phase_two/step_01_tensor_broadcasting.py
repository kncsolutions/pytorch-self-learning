# %% [markdown]
# Step 01 — Tensor Broadcasting
#
# Core intuition:
#
# Dimensions negotiate compatibility.
#
# Learn:
#
# singleton dimensions
# implicit expansion
# broadcasting rules
# shape alignment
#
# This is foundational.
# %%
import torch
# %% [markdown]
# ============================================================
# TENSOR BROADCASTING — GLOSSARY
# ============================================================

# This glossary introduces the major concepts,
# tensor operations, and mathematical ideas used in:
#
#     step_01_tensor_broadcasting.py
#
# The goal is not only API familiarity.
#
# The goal is to develop:
#
# - tensor intuition
# - dimensional reasoning
# - broadcasting geometry understanding

# %% [markdown]
# ============================================================
# WHAT IS BROADCASTING?
# ============================================================

# Broadcasting is PyTorch's mechanism for performing
# operations between tensors with different shapes.
#
# Instead of explicitly copying data,
# PyTorch attempts to:
#
#     align dimensions geometrically
#
# before computation.
#
# Example:
#
#     tensor_1.shape = [3, 4]
#     tensor_2.shape = [1, 4]
#
# PyTorch implicitly expands:
#
#     [1, 4]
#         ↓
#     [3, 4]
#
# This process is called:
#
#     broadcasting
# %%
print("="*80)
print("Tensor Broadcasting")
print("="*80)
x1 = torch.rand(3, 4)
y1 = torch.randn(1, 4)
print(f"x:\n {x1},\n y:\n {y1}")
# Observe how the shapes are different

# %% [markdown]


# ============================================================
# CORE BROADCASTING INTUITION
# ============================================================

# Broadcasting is not random expansion.
#
# It follows structural compatibility rules.
#
# Core idea:
#
#     Dimensions negotiate compatibility.

# %% [markdown]
# ============================================================
# IMPORTANT TENSOR CONCEPTS
# ============================================================

# %% [markdown]
# ============================================================
# TENSOR SHAPE
# ============================================================

# The shape of a tensor describes
# its dimensional structure.
#
# Example:
#
#     tensor.shape
#
# Output:
#
#     torch.Size([3, 4])
#
# Meaning:
#
# - 3 rows
# - 4 columns
# %% [markdown]
# ============================================================
# DIMENSION
# ============================================================

# A dimension is one axis of tensor structure.
#
# Example:
#
#     [Batch, Channels, Height, Width]
#
# contains:
#
# - Batch dimension
# - Channel dimension
# - Height dimension
# - Width dimension

# %% [markdown]
# ============================================================
# SINGLETON DIMENSION
# ============================================================

# A singleton dimension has size:
#
#     1
#
# Example:
#
#     [1, 4]
#
# Singleton dimensions are extremely important
# because they can expand during broadcasting.
# %%
print("shape of x1:", x1.shape)
print("shape of y:", y1.shape)
print(f" add(x,y):\n {torch.add(x1,y1)})")
print("."*80)
x2 = torch.rand(3, 4)
y2 = torch.randn(2, 4)
print(f"x:\n {x1},\n y:\n {y1}")
print("shape of x2:", x2.shape)
print("shape of y2:", y2.shape)
try:
    print(f" add(x2,y2):\n {torch.add(x2,y2)})")
except Exception as e:
    print("." * 80)
    print(f" add(x2,y2): Error: {e}")
    print("." * 80)
    print('CAN YOU SEE WHY YOU CAN ADD x1 AND y1 BUT YOU CANNOT ADD x2 AND y2?')
    print("." * 80)
    pass


# %% [markdown]
# ============================================================
# BROADCASTING RULES
# ============================================================

# PyTorch compares tensor shapes:
#
#     right to left
#
# Two dimensions are compatible if:
#
# 1. They are equal
#
# OR
#
# 2. One of them is:
#
#     1

# %% [markdown]
# ============================================================
# EXAMPLE BROADCASTING
# ============================================================

# Tensor A:
#
#     [3, 4]
#
# Tensor B:
#
#     [1, 4]
#
# Broadcasting result:
#
#     [3, 4]
#
# because:
#
#     1 expands to 3

# %% [markdown]
# ============================================================
# TENSOR OPERATIONS USED
# ============================================================

# %% [markdown]
# ============================================================
# ADDITION
# ============================================================

# torch.add()
#
# or
#
# +
#
# Performs element-wise addition.
#
# Example:
#
#     a + b

# %% [markdown]
# ============================================================
# MULTIPLICATION
# ============================================================

# torch.mul()
#
# or
#
# *
#
# Performs element-wise multiplication.

# %% [markdown]
# ============================================================
# SHAPE INSPECTION
# ============================================================

# tensor.shape
#
# Returns tensor dimensions.

# %% [markdown]
# ============================================================
# DIMENSION EXPANSION
# ============================================================

# tensor.unsqueeze()
#
# Adds a new dimension.
#
# Example:
#
#     tensor.unsqueeze(0)

# %% [markdown]
# ============================================================
# DIMENSION REMOVAL
# ============================================================

# tensor.squeeze()
#
# Removes singleton dimensions.

# %% [markdown]
# ============================================================
# TENSOR RESHAPING
# ============================================================

# tensor.reshape()
#
# Changes tensor geometry
# without changing data.

# %% [markdown]
# ============================================================
# TENSOR VIEWING
# ============================================================

# tensor.view()
#
# Creates a different tensor
# perspective over memory.

# %% [markdown]
# ============================================================
# EXPLICIT EXPANSION
# ============================================================

# tensor.expand()
#
# Explicitly expands singleton dimensions
# without copying memory.
#
# Example:
#
#     tensor.expand(3, 4)

# %% [markdown]
# ============================================================
# TENSOR REPETITION
# ============================================================

# tensor.repeat()
#
# Repeats tensor data physically in memory.
#
# Unlike broadcasting:
#
#     repeat() copies data
#     expand() does not

# %% [markdown]
# ============================================================
# MATHEMATICAL CONCEPTS
# ============================================================

# %% [markdown]
# ============================================================
# ELEMENT-WISE OPERATIONS
# ============================================================

# Operations are applied independently
# to corresponding tensor elements.
#
# Example:
#
#     A[i, j] + B[i, j]

# %% [markdown]
# ============================================================
# SHAPE COMPATIBILITY
# ============================================================

# Broadcasting requires compatible tensor geometry.
#
# Mathematically:
#
#     dimensions must align structurally
#
# before operations can occur.

# %% [markdown]
# ============================================================
# IMPLICIT EXPANSION
# ============================================================

# Broadcasting creates the illusion
# of larger tensors without copying memory.
#
# This is conceptually similar to:
#
#     virtual geometric expansion

# %% [markdown]
# ============================================================
# TENSOR GEOMETRY
# ============================================================

# Tensor shapes encode structural meaning.
#
# Example:
#
#     [Batch, Channels, Height, Width]
#
# is not merely:
#
#     four numbers
#
# It represents:
#
# - tensor population
# - information fields
# - spatial structure

# %% [markdown]
# ============================================================
# COMMON BROADCASTING PATTERNS
# ============================================================

# %% [markdown]
# ============================================================
# VECTOR + SCALAR
# ============================================================

# [3] + [1]
#
# Example:
#
#     tensor + 5
#
# The scalar broadcasts across all elements.

# %% [markdown]
# ============================================================
# MATRIX + VECTOR
# ============================================================

# [3, 4] + [4]
#
# The vector broadcasts across rows.

# %% [markdown]
# ============================================================
# BATCH BROADCASTING
# ============================================================

# [32, 3, 224, 224]
# +
# [1, 3, 1, 1]
#
# Common in image normalization.

# %% [markdown]
# ============================================================
# IMPORTANT DEEP LEARNING CONNECTION
# ============================================================

# Broadcasting appears constantly in:
#
# - normalization
# - attention
# - embeddings
# - masking
# - convolution systems
# - transformer architectures
#
# Understanding broadcasting deeply
# is foundational for modern AI systems.

# %% [markdown]
# ============================================================
# MOST IMPORTANT INTUITION
# ============================================================

# Broadcasting is not merely:
#
#     automatic expansion
#
# It is:
#
#     structured tensor alignment
#
# between compatible geometries.

# %% [markdown]
# ============================================================
# VECTOR + SCALAR BROADCASTING
# ============================================================

# A scalar can broadcast across
# every element of a tensor.
# %%

print("\n" + "=" * 80)
print("VECTOR + SCALAR BROADCASTING")
print("=" * 80)

vector = torch.tensor([1, 2, 3])

scalar = 10

print(f"vector:\n{vector}")

print(f"\nscalar:\n{scalar}")

result = vector + scalar

print(f"\nresult:\n{result}")

print("\nObservation:")

print(
    "The scalar value expands implicitly "
    "across every tensor element."
)

# %%
# ============================================================
# MATRIX + VECTOR BROADCASTING
# ============================================================

# Here the vector broadcasts across rows.

print("\n" + "=" * 80)
print("MATRIX + VECTOR BROADCASTING")
print("=" * 80)

matrix = torch.tensor(
    [
        [1, 2, 3],
        [4, 5, 6]
    ]
)

vector = torch.tensor([10, 20, 30])

print(f"matrix shape : {matrix.shape}")

print(f"vector shape : {vector.shape}")

print(f"\nmatrix:\n{matrix}")

print(f"\nvector:\n{vector}")

result = matrix + vector

print(f"\nresult:\n{result}")

print("\nObservation:")

print(
    "The vector broadcasts across "
    "each row of the matrix."
)

# %%
# ============================================================
# UNSQUEEZE FOR BROADCASTING
# ============================================================

# unsqueeze() adds a singleton dimension.

print("\n" + "=" * 80)
print("UNSQUEEZE FOR BROADCASTING")
print("=" * 80)

tensor = torch.tensor([1, 2, 3])

print(f"original tensor shape : {tensor.shape}")

unsqueezed_tensor = tensor.unsqueeze(0)

print(
    f"unsqueezed tensor shape : "
    f"{unsqueezed_tensor.shape}"
)

print(f"\nunsqueezed tensor:\n{unsqueezed_tensor}")

print("\nObservation:")

print(
    "unsqueeze() introduces a new dimension "
    "without changing tensor values."
)

# %%
# ============================================================
# COLUMN-WISE BROADCASTING
# ============================================================

# Broadcasting can also occur across columns.

print("\n" + "=" * 80)
print("COLUMN-WISE BROADCASTING")
print("=" * 80)

matrix = torch.tensor(
    [
        [1, 2, 3],
        [4, 5, 6]
    ]
)

column_vector = torch.tensor(
    [
        [10],
        [20]
    ]
)

print(f"matrix shape        : {matrix.shape}")

print(
    f"column vector shape : "
    f"{column_vector.shape}"
)

print(f"\nmatrix:\n{matrix}")

print(f"\ncolumn vector:\n{column_vector}")

result = matrix + column_vector

print(f"\nresult:\n{result}")

print("\nObservation:")

print(
    "The singleton dimension expands "
    "across columns."
)

# %%
# ============================================================
# EXPLICIT EXPANSION
# ============================================================

# expand() creates a broadcasted view
# without copying memory.

print("\n" + "=" * 80)
print("EXPLICIT EXPANSION")
print("=" * 80)

tensor = torch.tensor([[1, 2, 3]])

print(f"original shape : {tensor.shape}")

expanded_tensor = tensor.expand(4, 3)

print(f"expanded shape : {expanded_tensor.shape}")

print(f"\nexpanded tensor:\n{expanded_tensor}")

print("\nObservation:")

print(
    "expand() creates virtual expansion "
    "without physically copying tensor data."
)

# %%
# ============================================================
# REPEAT VS EXPAND
# ============================================================

# repeat() physically copies memory.

print("\n" + "=" * 80)
print("REPEAT VS EXPAND")
print("=" * 80)

tensor = torch.tensor([[1, 2, 3]])

expanded = tensor.expand(4, 3)

repeated = tensor.repeat(4, 1)

print(f"expanded shape : {expanded.shape}")

print(f"repeated shape : {repeated.shape}")

print(f"\nexpanded tensor:\n{expanded}")

print(f"\nrepeated tensor:\n{repeated}")

print("\nObservation:")

print(
    "expand() creates virtual broadcasting.\n"
    "repeat() physically duplicates data."
)

# %%
# ============================================================
# MEMORY COMPARISON
# ============================================================

print("\n" + "=" * 80)
print("MEMORY COMPARISON")
print("=" * 80)

print(
    f"Original Storage Size : "
    f"{tensor.storage().size()}"
)

print(
    f"Expanded Storage Size : "
    f"{expanded.storage().size()}"
)

print(
    f"Repeated Storage Size : "
    f"{repeated.storage().size()}"
)

print("\nObservation:")

print(
    "expand() reuses memory.\n"
    "repeat() allocates additional memory."
)

# %%
# ============================================================
# HIGHER DIMENSION BROADCASTING
# ============================================================

# Broadcasting becomes extremely important
# in deep learning systems.

print("\n" + "=" * 80)
print("HIGHER DIMENSION BROADCASTING")
print("=" * 80)

batch_tensor = torch.randn(
    2, 3, 4
)

normalization_tensor = torch.randn(
    1, 3, 1
)

print(
    f"batch tensor shape : "
    f"{batch_tensor.shape}"
)

print(
    f"normalization tensor shape : "
    f"{normalization_tensor.shape}"
)

result = batch_tensor + normalization_tensor

print(f"\nresult shape : {result.shape}")

print("\nObservation:")

print(
    "Singleton dimensions broadcast across:\n"
    "- batch dimension\n"
    "- spatial dimension"
)

# %%
# ============================================================
# BROADCASTING FAILURE EXAMPLE
# ============================================================

print("\n" + "=" * 80)
print("BROADCASTING FAILURE EXAMPLE")
print("=" * 80)

a = torch.randn(2, 3)

b = torch.randn(4, 3)

print(f"a shape : {a.shape}")

print(f"b shape : {b.shape}")

try:

    result = a + b

except Exception as error:

    print("\nBroadcasting Failed.")

    print(f"\nError:\n{error}")

    print("\nObservation:")

    print(
        "Dimensions are incompatible because:\n"
        "2 cannot broadcast with 4"
    )

# %%
# ============================================================
# BROADCASTING VISUALIZATION
# ============================================================

print("\n" + "=" * 80)
print("BROADCASTING VISUALIZATION")
print("=" * 80)

print(
    """
Example:

    Tensor A : [3, 4]
    Tensor B : [1, 4]

Alignment:

        [3, 4]
        [1, 4]
         ↓
        [3, 4]

The singleton dimension expands
to match compatible geometry.
"""
)

# %% 
# ============================================================
# FINAL CONCEPTUAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FINAL CONCEPTUAL SUMMARY")
print("=" * 80)

print(
    """
Broadcasting is one of the most important
ideas in tensor systems.

It enables:

- efficient tensor interaction
- implicit geometric alignment
- scalable deep learning computation

Broadcasting is not merely:

    automatic expansion

It is:

    structured tensor compatibility negotiation
"""
)