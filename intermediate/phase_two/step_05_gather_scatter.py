# %% [markdown]
"""
# STEP 05 — GATHER AND SCATTER

## CORE INTUITION

Gather and scatter are tensor routing operations.

They allow tensor systems to:

- collect information
- redistribute information
- dynamically route values
- reorganize tensor structure

Core idea:

    tensors can move information selectively

This becomes extremely important in:

- attention systems
- transformers
- graph neural networks
- sparse computation
- recommendation systems
- reinforcement learning

This step introduces:

- gather()
- scatter()
- index-based routing
- tensor information flow
- dynamic tensor interaction
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

## Gather

gather() extracts tensor values
using index tensors.


------------------------------------------------------------

## Scatter

scatter() writes tensor values
into selected tensor locations.


------------------------------------------------------------

## Index Tensor

An index tensor specifies:

    where information should move.


------------------------------------------------------------

## Tensor Routing

Gather and scatter enable dynamic
information routing inside tensor systems.


------------------------------------------------------------

## Sparse Interaction

These operations allow selective computation
without processing entire tensors.
"""
# %%


# %% [markdown]
"""
# ============================================================
# SECTION 01 — BASIC GATHER
# ============================================================

gather() extracts tensor values
from specified index positions.
"""
# %%


# ============================================================
# BASIC GATHER
# ============================================================

tensor = torch.tensor(
    [
        [10, 20, 30],
        [40, 50, 60]
    ]
)

index_tensor = torch.tensor(
    [
        [0, 2],
        [1, 0]
    ]
)

print("\n" + "=" * 80)
print("BASIC GATHER")
print("=" * 80)

print(f"\nOriginal Tensor:\n{tensor}")

print(f"\nIndex Tensor:\n{index_tensor}")

result = torch.gather(
    tensor,
    dim=1,
    index=index_tensor
)

print(f"\nGather Result:\n{result}")


# %% [markdown]
"""
# GATHER GEOMETRY

dim=1 means:

    gather across columns

Selections:

    tensor[0,0] -> 10
    tensor[0,2] -> 30

    tensor[1,1] -> 50
    tensor[1,0] -> 40
"""
# %%


# %% [markdown]
"""
# ============================================================
# SECTION 02 — GATHER ALONG ROWS
# ============================================================

Gathering can occur across different dimensions.
"""
# %%


# ============================================================
# ROW-WISE GATHER
# ============================================================

tensor = torch.tensor(
    [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ]
)

index_tensor = torch.tensor(
    [
        [0, 1, 2],
        [2, 1, 0]
    ]
)

print("\n" + "=" * 80)
print("ROW-WISE GATHER")
print("=" * 80)

print(f"\nOriginal Tensor:\n{tensor}")

print(f"\nIndex Tensor:\n{index_tensor}")

result = torch.gather(
    tensor,
    dim=0,
    index=index_tensor
)

print(f"\nGather Result:\n{result}")


# %% [markdown]
"""
# DIMENSIONAL INTERPRETATION

dim=0 means:

    gather across rows

The index tensor specifies:

    which row to extract from
"""
# %%


# %% [markdown]
"""
# ============================================================
# SECTION 03 — GATHER AND PROBABILITY SELECTION
# ============================================================

Gather is heavily used in classification systems.
"""
# %%


# ============================================================
# CLASS PROBABILITY SELECTION
# ============================================================

probabilities = torch.tensor(
    [
        [0.1, 0.7, 0.2],
        [0.8, 0.1, 0.1],
        [0.2, 0.3, 0.5]
    ]
)

target_classes = torch.tensor(
    [
        [1],
        [0],
        [2]
    ]
)

print("\n" + "=" * 80)
print("CLASS PROBABILITY GATHER")
print("=" * 80)

print(f"\nProbabilities:\n{probabilities}")

print(f"\nTarget Classes:\n{target_classes}")

selected_probabilities = torch.gather(
    probabilities,
    dim=1,
    index=target_classes
)

print(
    f"\nSelected Probabilities:\n"
    f"{selected_probabilities}"
)


# %% [markdown]
"""
# DEEP LEARNING CONNECTION

Cross entropy systems often gather:

    correct class probabilities

using target labels as index tensors.
"""
# %%


# %% [markdown]
"""
# ============================================================
# SECTION 04 — BASIC SCATTER
# ============================================================

scatter() writes values into selected positions.
"""
# %%


# ============================================================
# BASIC SCATTER
# ============================================================

base_tensor = torch.zeros(
    3, 5
)

index_tensor = torch.tensor(
    [
        [0, 2],
        [1, 3],
        [4, 0]
    ]
)

source_tensor = torch.tensor(
    [
        [10, 20],
        [30, 40],
        [50, 60]
    ]
).float()

print("\n" + "=" * 80)
print("BASIC SCATTER")
print("=" * 80)

print(f"\nBase Tensor:\n{base_tensor}")

print(f"\nIndex Tensor:\n{index_tensor}")

print(f"\nSource Tensor:\n{source_tensor}")

result = base_tensor.scatter(
    dim=1,
    index=index_tensor,
    src=source_tensor
)

print(f"\nScatter Result:\n{result}")


# %% [markdown]
"""
# SCATTER GEOMETRY

scatter():

    routes source values into target positions

The index tensor determines:

    where information is written.
"""
# %%


# %% [markdown]
"""
# ============================================================
# SECTION 05 — SCATTER WITH CONSTANT VALUES
# ============================================================

scatter() can also insert constant values.
"""
# %%


# ============================================================
# CONSTANT VALUE SCATTER
# ============================================================

tensor = torch.zeros(3, 4)

index_tensor = torch.tensor(
    [
        [1],
        [2],
        [0]
    ]
)

print("\n" + "=" * 80)
print("CONSTANT VALUE SCATTER")
print("=" * 80)

print(f"\nOriginal Tensor:\n{tensor}")

result = tensor.scatter(
    dim=1,
    index=index_tensor,
    value=1
)

print(f"\nScatter Result:\n{result}")


# %% [markdown]
"""
# GEOMETRIC INTERPRETATION

The value:

    1

is routed into indexed positions.
"""
# %%


# %% [markdown]
"""
# ============================================================
# SECTION 06 — ONE-HOT ENCODING
# ============================================================

scatter() is commonly used to create
one-hot representations.
"""
# %%


# ============================================================
# ONE-HOT ENCODING
# ============================================================

class_indices = torch.tensor(
    [
        [0],
        [2],
        [1]
    ]
)

num_classes = 4

one_hot = torch.zeros(
    class_indices.size(0),
    num_classes
)

print("\n" + "=" * 80)
print("ONE-HOT ENCODING")
print("=" * 80)

print(f"\nClass Indices:\n{class_indices}")

one_hot.scatter_(
    dim=1,
    index=class_indices,
    value=1
)

print(f"\nOne-Hot Tensor:\n{one_hot}")


# %% [markdown]
"""
# ONE-HOT GEOMETRY

Each row becomes:

    a sparse class representation

Only one position contains:

    1
"""
# %%


# %% [markdown]
"""
# ============================================================
# SECTION 07 — HIGHER DIMENSION GATHER
# ============================================================

Gather also works on higher-dimensional tensors.
"""
# %%


# ============================================================
# HIGHER DIMENSION GATHER
# ============================================================

tensor = torch.randn(
    2, 3, 4
)

index_tensor = torch.tensor(
    [
        [
            [0, 1],
            [2, 0],
            [1, 3]
        ],

        [
            [3, 2],
            [1, 0],
            [0, 2]
        ]
    ]
)

print("\n" + "=" * 80)
print("HIGHER DIMENSION GATHER")
print("=" * 80)

print(f"\nTensor Shape : {tensor.shape}")

print(f"Index Shape  : {index_tensor.shape}")

result = torch.gather(
    tensor,
    dim=2,
    index=index_tensor
)

print(f"\nResult Shape : {result.shape}")


# %% [markdown]
"""
# HIGHER DIMENSION INTERPRETATION

Gather operates independently across
specified dimensions.

This enables dynamic information extraction
inside complex tensor systems.
"""
# %%


# %% [markdown]
"""
# ============================================================
# SECTION 08 — ATTENTION INTUITION
# ============================================================

Transformers heavily rely on tensor routing.
"""
# %%


# ============================================================
# SIMPLE ATTENTION ROUTING
# ============================================================

attention_scores = torch.tensor(
    [
        [0.1, 0.8, 0.1],
        [0.6, 0.2, 0.2]
    ]
)

top_indices = torch.argmax(
    attention_scores,
    dim=1,
    keepdim=True
)

values = torch.tensor(
    [
        [100, 200, 300],
        [400, 500, 600]
    ]
)

print("\n" + "=" * 80)
print("ATTENTION ROUTING")
print("=" * 80)

print(f"\nAttention Scores:\n{attention_scores}")

print(f"\nTop Indices:\n{top_indices}")

selected_values = torch.gather(
    values,
    dim=1,
    index=top_indices
)

print(f"\nSelected Values:\n{selected_values}")


# %% [markdown]
"""
# ATTENTION CONNECTION

Attention systems dynamically gather:

    relevant information

based on learned interaction scores.
"""
# %%


# %% [markdown]
"""
# ============================================================
# SECTION 09 — MEMORY AND ROUTING
# ============================================================

Gather and scatter manipulate tensor flow
without necessarily modifying original geometry.
"""
# %%


# ============================================================
# MEMORY INFORMATION
# ============================================================

tensor = torch.randn(1000, 1000)

index_tensor = torch.randint(
    0,
    1000,
    (1000, 100)
)

print("\n" + "=" * 80)
print("MEMORY INFORMATION")
print("=" * 80)

print(
    f"\nTensor Memory : "
    f"{tensor.numel() * tensor.element_size()} Bytes"
)

print(
    f"Index Memory  : "
    f"{index_tensor.numel() * index_tensor.element_size()} Bytes"
)


# %% [markdown]
"""
# MEMORY OBSERVATION

Routing systems require index tensors.

Large-scale routing operations can become
memory intensive in advanced architectures.
"""
# %%


# %% [markdown]
"""
# ============================================================
# IMPORTANT DEEP LEARNING CONNECTIONS
# ============================================================

Gather and scatter appear constantly in:

- transformers
- graph neural networks
- sparse systems
- recommendation systems
- reinforcement learning
- token routing
- mixture-of-experts systems
- attention architectures

Understanding tensor routing deeply is foundational
for modern AI systems.
"""
# %%


# %% [markdown]
"""
# ============================================================
# FINAL CONCEPTUAL SUMMARY
# ============================================================

Gather and scatter enable dynamic tensor routing.

They allow tensor systems to:

- extract information
- redirect information
- build sparse interaction systems
- perform selective computation

Gather and scatter are not merely:

    indexing operations

They are:

    structured tensor information routing systems.
"""
# %%