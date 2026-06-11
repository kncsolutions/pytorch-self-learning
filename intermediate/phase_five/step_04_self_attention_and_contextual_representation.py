#%%  [markdown]
"""
# ============================================================
# STEP 04 — Self-Attention and Contextual Representation
# ============================================================

# Core Idea

Representations can dynamically reinterpret each other.

One of the deepest breakthroughs in modern AI systems:

the meaning or representation of an entity can change
depending on surrounding context.

This process is called:

        self-attention

------------------------------------------------------------
# Central Philosophy
------------------------------------------------------------

Self-attention is NOT merely:

        "words looking at words"

Instead it is:

        dynamic contextual interaction
between representations.

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Representations are NOT always static.

Context can reshape:

    * interpretation
    * relevance
    * information flow
    * representation geometry

------------------------------------------------------------
# Self-Attention Intuition
------------------------------------------------------------

Each representation simultaneously acts as:

    * query
    * key
    * value

This allows representations to:

        dynamically interact with each other.

------------------------------------------------------------
# Mathematical Formulation
------------------------------------------------------------
"""


#%%  [markdown]
"""
where:

    X = input representations
    Q = query matrix
    K = key matrix
    V = value matrix

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Self-attention creates:

        contextual representation transformation

------------------------------------------------------------
# Learning Goals
------------------------------------------------------------

By the end of this file you should understand:

    * self-attention
    * contextual representation
    * dynamic reinterpretation
    * attention interaction matrices
    * contextual aggregation
    * information blending
    * relational geometry

------------------------------------------------------------
# Glossary of Important Functions
------------------------------------------------------------

torch.matmul(A, B)
    Matrix multiplication.

torch.softmax(x, dim)
    Converts scores into normalized attention weights.

torch.sum(x, dim)
    Aggregates weighted information.

unsqueeze(dim)
    Adds an extra dimension for broadcasting.

transpose(dim0, dim1)
    Swaps tensor dimensions.

------------------------------------------------------------
# Glossary of Mathematical Concepts
------------------------------------------------------------

Dot Product:
        Measures vector alignment.

Softmax:
        Converts interaction scores into normalized weights.

Attention Matrix:
        Stores pairwise interaction strengths.

Contextual Representation:
        Representation updated using surrounding context.

------------------------------------------------------------
"""

#%%
import torch

torch.manual_seed(42)

#%%  [markdown]
"""
# ============================================================
# SECTION 1 — INPUT REPRESENTATIONS
# ============================================================

Suppose we have a sequence of representations.

These may represent:

    * words
    * image patches
    * latent states
    * semantic embeddings

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

In self-attention:

        representations interact with each other.
"""

#%%
tokens = [
    "animal",
    "tree",
    "cat",
    "river"
]

X = torch.tensor([
    [0.9, 0.8],    # animal
    [0.1, 0.2],    # tree
    [0.85, 0.75],  # cat
    [-0.8, -0.7]   # river
])

print("\nInput Representations:\n")
print(X)

print("\nTensor Shape:")
print(X.shape)

#%%  [markdown]
"""
Tensor shape:

            (4, 2)

means:

    4 representations
    2-dimensional representation space
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 2 — QUERY, KEY, VALUE PROJECTIONS
# ============================================================

Self-attention creates:

    * queries
    * keys
    * values

from the same input representations.

------------------------------------------------------------
# Mathematical Formulation
------------------------------------------------------------

Q = XW_Q
K = XW_K
V = XW_V

where:

    W_Q = query projection matrix
    W_K = key projection matrix
    W_V = value projection matrix

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

The same representations generate:

        interaction roles
"""

#%%
W_Q = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0]
])

W_K = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0]
])

W_V = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0]
])

Q = torch.matmul(X, W_Q)
K = torch.matmul(X, W_K)
V = torch.matmul(X, W_V)

print("\nQuery Matrix:\n")
print(Q)

print("\nKey Matrix:\n")
print(K)

print("\nValue Matrix:\n")
print(V)

#%%  [markdown]
"""
Observe:

For simplicity:

        projection matrices are identity matrices.

So:

        Q ≈ K ≈ V ≈ X

------------------------------------------------------------
# Important Clarification
------------------------------------------------------------

Real transformers learn these projection matrices.
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 3 — RELATIONAL MATCHING
# ============================================================

Self-attention computes pairwise compatibility
between representations.

------------------------------------------------------------
# Mathematical Formulation
------------------------------------------------------------

Compatibility matrix:

            S = QK^T

------------------------------------------------------------
# Geometric Interpretation
------------------------------------------------------------

Large score:
        strong relational alignment

Small score:
        weak relational alignment

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Representations dynamically evaluate
their relevance to each other.
"""

#%%
scores = torch.matmul(
    Q,
    K.transpose(0, 1)
)

print("\nCompatibility Score Matrix:\n")
print(scores)

#%%  [markdown]
"""
Interpretation:

Row i:
        query representation

Column j:
        key representation

Each value measures:

        relational compatibility
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 4 — SCALED ATTENTION SCORES
# ============================================================

Attention scores are scaled for numerical stability.

------------------------------------------------------------
# Mathematical Formulation
------------------------------------------------------------

Scaled scores:

                S / sqrt(d_k)

where:

    d_k = key dimension

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Scaling prevents excessively large interaction values.
"""

#%%
d_k = K.shape[1]

scaled_scores = scores / torch.sqrt(
    torch.tensor(float(d_k))
)

print("\nScaled Attention Scores:\n")
print(scaled_scores)

#%%  [markdown]
"""
Observe:

Scaling reduced score magnitude.

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Stable geometry becomes important
for deep representation systems.
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 5 — ATTENTION WEIGHT MATRIX
# ============================================================

Softmax converts scores into:

        attention weights

------------------------------------------------------------
# Mathematical Formulation
------------------------------------------------------------
"""


#%%  [markdown]
"""
------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Attention weights determine:

        interaction influence strength
"""

#%%
attention_matrix = torch.softmax(
    scaled_scores,
    dim=1
)

print("\nAttention Weight Matrix:\n")
print(attention_matrix)

#%%  [markdown]
"""
Interpretation:

Each row represents:

        how strongly one representation
        attends to all others.

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Attention creates:

        dynamic interaction topology
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 6 — CONTEXTUAL REPRESENTATION AGGREGATION
# ============================================================

Self-attention updates representations
through weighted contextual aggregation.

------------------------------------------------------------
# Mathematical Formulation
------------------------------------------------------------

Contextual output:

                Y = AV

where:

    A = attention matrix
    V = value matrix

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Representations become mixtures of:

        contextual information
"""

#%%
contextual_representations = torch.matmul(
    attention_matrix,
    V
)

print("\nContextual Representations:\n")
print(contextual_representations)

#%%  [markdown]
"""
Observe:

Each representation was updated using:

        information from other representations

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Self-attention enables:

        contextual reinterpretation
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 7 — REPRESENTATION TRANSFORMATION
# ============================================================

Compare:

    original representations
            vs
    contextual representations

------------------------------------------------------------
# Important Concept
------------------------------------------------------------

Self-attention transforms representations dynamically.

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Meaning becomes:

        context-dependent geometry
"""

#%%
print("\nOriginal Representations:\n")
print(X)

print("\nUpdated Contextual Representations:\n")
print(contextual_representations)

#%%  [markdown]
"""
Observe:

Representations changed after contextual interaction.

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Self-attention allows representations to:

        reshape each other dynamically
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 8 — INFORMATION FLOW INTERPRETATION
# ============================================================

Attention matrices simulate:

        information flow networks

------------------------------------------------------------
# Important Concept
------------------------------------------------------------

Strong attention pathway:
        strong information transfer

Weak attention pathway:
        weak information transfer

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Self-attention dynamically controls:

        representational influence propagation
"""

#%%
print("\nInformation Flow Strengths:\n")

for i in range(len(tokens)):

    print(f"\n{tokens[i]} attends to:\n")

    for j in range(len(tokens)):

        print(
            f"    {tokens[j]}"
            f" : {attention_matrix[i][j]:.4f}"
        )

#%%  [markdown]
"""
Observe:

Each representation distributes attention differently.

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Self-attention creates:

        adaptive contextual interaction systems
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 9 — CONTEXTUAL MEANING
# ============================================================

One of the deepest ideas in transformers:

representations become context-sensitive.

------------------------------------------------------------
# Example Intuition
------------------------------------------------------------

The meaning of:

        "bank"

depends on surrounding context.

Examples:

    * river bank
    * financial bank

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Self-attention enables:

        contextual reinterpretation

inside representation geometry.
"""

#%%
bank_context_1 = torch.tensor([
    [0.9, 0.8],    # river
    [0.85, 0.75],  # bank
])

bank_context_2 = torch.tensor([
    [-0.8, -0.7],  # money
    [-0.75, -0.65] # bank
])

print("\nRiver Bank Context:\n")
print(bank_context_1)

print("\nFinancial Bank Context:\n")
print(bank_context_2)

#%%  [markdown]
"""
Observe:

Context changes relational positioning.

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Representations gain meaning through:

        contextual interaction geometry
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 10 — SELF-ATTENTION IN MODERN AI
# ============================================================

Self-attention became one of the foundations
of modern AI systems.

------------------------------------------------------------
# Applications
------------------------------------------------------------

Self-attention is central for:

    * transformers
    * language models
    * vision transformers
    * multimodal systems
    * retrieval systems
    * generative AI

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Modern AI frequently relies on:

        contextual representation interaction
"""

#%%
applications = [
    "Transformers",
    "Large Language Models",
    "Vision Transformers",
    "Multimodal AI",
    "Semantic Retrieval",
    "Generative Systems"
]

print("\nApplications of Self-Attention:\n")

for application in applications:

    print(f"- {application}")

#%%  [markdown]
"""
# ============================================================
# SECTION 11 — REPRESENTATION INTERACTION PIPELINE
# ============================================================

One of the deepest conceptual shifts in modern AI:

representations dynamically reinterpret each other.

------------------------------------------------------------
# Representation Flow
------------------------------------------------------------

            Representations
                    ↓
             Query / Key Matching
                    ↓
             Attention Weights
                    ↓
           Contextual Aggregation
                    ↓
        Updated Representations
                    ↓
        Contextual Meaning Formation

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Self-attention continuously reshapes:

        contextual representation geometry
"""

#%%
print("\nSelf-Attention Pipeline:\n")

print("Representations")
print("       ↓")
print("Query-Key Matching")
print("       ↓")
print("Attention Matrix")
print("       ↓")
print("Weighted Aggregation")
print("       ↓")
print("Contextual Representations")
print("       ↓")
print("Dynamic Meaning")

#%%  [markdown]
"""
# ============================================================
# FINAL SUMMARY
# ============================================================

# Key Ideas Learned

1. Self-attention allows representations to interact dynamically

2. Queries, keys, and values emerge from the same representations

3. Dot-product matching measures relational compatibility

4. Attention matrices determine interaction strength

5. Self-attention creates contextual representations

6. Representations become context-dependent

7. Modern AI systems frequently rely on contextual geometry

------------------------------------------------------------
# Most Important Insight
------------------------------------------------------------

Self-attention enables representations to:

        dynamically reinterpret each other

through:

        contextual interaction geometry

inside representation space.

Meaning becomes:

        context-sensitive relational structure.

------------------------------------------------------------
# Connection to Future Topics
------------------------------------------------------------

This becomes foundational for:

    * transformers
    * language models
    * multimodal AI
    * generative systems
    * contextual reasoning
"""