#%%  [markdown]
"""
# ============================================================
# STEP 02 — Queries, Keys, and Relational Matching
# ============================================================

# Core Idea

Attention measures relational compatibility.

One of the deepest ideas in modern attention systems:

representations dynamically determine
which other representations are relevant.

This relevance is computed through:

        relational matching

between vectors.

------------------------------------------------------------
# Central Philosophy
------------------------------------------------------------

Attention is fundamentally:

        similarity-driven interaction selection

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Queries and keys create a mechanism for:

        dynamic relational comparison

inside representation space.

------------------------------------------------------------
# Core Components
------------------------------------------------------------

Attention systems typically use:

    * Query vectors (Q)
    * Key vectors   (K)
    * Value vectors (V)

------------------------------------------------------------
# Intuition
------------------------------------------------------------

Query:
        "What information am I searching for?"

Key:
        "What kind of information do I contain?"

Value:
        "What information should be transferred?"

------------------------------------------------------------
# Mathematical Formulation
------------------------------------------------------------
"""

#%%  [markdown]
"""
where:

    Q = query matrix
    K = key matrix
    V = value matrix

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Attention computes:

        relational compatibility

between representations.

------------------------------------------------------------
# Learning Goals
------------------------------------------------------------

By the end of this file you should understand:

    * queries
    * keys
    * values
    * relational matching
    * dot-product similarity
    * compatibility scores
    * dynamic interaction selection

------------------------------------------------------------
"""

#%%
import torch

torch.manual_seed(42)

#%%  [markdown]
"""
# ============================================================
# SECTION 1 — REPRESENTATION VECTORS
# ============================================================

Suppose we already have representation vectors.

These may represent:

    * words
    * image patches
    * latent states
    * semantic embeddings

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Attention systems compare representations
through relational geometry.
"""

#%%
representations = torch.tensor([
    [0.9, 0.8],    # token 1
    [0.1, 0.2],    # token 2
    [0.85, 0.75],  # token 3
    [-0.8, -0.7]   # token 4
])

print("\nRepresentation Vectors:\n")
print(representations)

#%%  [markdown]
"""
Each row becomes:

        one representation vector

inside representation space.
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 2 — QUERY VECTORS
# ============================================================

A query vector represents:

        what information is being searched for

------------------------------------------------------------
# Intuition
------------------------------------------------------------

The query asks:

        "Which representations are relevant to me?"

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Queries define interaction intent.
"""

#%%
query = torch.tensor([
    0.88,
    0.79
])

print("\nQuery Vector:\n")
print(query)

#%%  [markdown]
"""
Observe:

The query resembles:

    token 1
    token 3

This suggests those tokens may become relevant.
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 3 — KEY VECTORS
# ============================================================

Key vectors represent:

        searchable representation signatures

------------------------------------------------------------
# Intuition
------------------------------------------------------------

A key answers:

        "What kind of information do I contain?"

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Keys enable relational matching.
"""

#%%
keys = representations

print("\nKey Vectors:\n")
print(keys)

#%%  [markdown]
"""
In many simplified examples:

        keys ≈ representations

But in real transformers:

    queries, keys, and values are often
    learned projections.
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 4 — DOT PRODUCT MATCHING
# ============================================================

Attention compares queries and keys using:

        dot-product interaction

------------------------------------------------------------
# Mathematical Formulation
------------------------------------------------------------

Dot product:

            Q · K

Expanded:

        Σ(q_i k_i)

------------------------------------------------------------
# Geometric Interpretation
------------------------------------------------------------

Large dot product:
        strong alignment

Small dot product:
        weak alignment

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Dot products measure:

        relational compatibility
"""

#%%
compatibility_scores = torch.matmul(
    keys,
    query
)

print("\nCompatibility Scores:\n")
print(compatibility_scores)

#%%  [markdown]
"""
Observe:

Some keys produced larger scores.

These representations align more strongly
with the query.

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Attention becomes:

        similarity-driven interaction selection
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 5 — WHY SCALING IS NECESSARY
# ============================================================

Large vector dimensions may produce:

        extremely large dot products

------------------------------------------------------------
# Mathematical Scaling
------------------------------------------------------------

Scaled attention divides by:

                sqrt(d_k)

where:

    d_k = key dimension

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Scaling stabilizes interaction magnitude.
"""

#%%
d_k = query.shape[0]

scaled_scores = compatibility_scores / torch.sqrt(
    torch.tensor(float(d_k))
)

print("\nScaled Compatibility Scores:\n")
print(scaled_scores)

#%%  [markdown]
"""
Observe:

Scaling reduced score magnitude.

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Stable numerical behavior becomes important
for deep attention systems.
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 6 — SOFTMAX AS RELATIONAL NORMALIZATION
# ============================================================

Raw compatibility scores are converted into:

        attention weights

using softmax.

------------------------------------------------------------
# Mathematical Formulation
------------------------------------------------------------
"""


#%%  [markdown]
"""
------------------------------------------------------------
# Geometric Interpretation
------------------------------------------------------------

Softmax converts scores into:

        normalized influence strengths

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Attention weights determine:

        interaction importance
"""

#%%
attention_weights = torch.softmax(
    scaled_scores,
    dim=0
)

print("\nAttention Weights:\n")
print(attention_weights)

print("\nSum of Attention Weights:")
print(attention_weights.sum())

#%%  [markdown]
"""
Observe:

Weights now form a probability-like distribution.

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Attention redistributes representational influence.
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 7 — VALUE VECTORS
# ============================================================

Value vectors contain:

        transferable information

------------------------------------------------------------
# Intuition
------------------------------------------------------------

Queries and keys decide:

        what is relevant

Values determine:

        what information flows

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Attention separates:

    * interaction matching
    * information transfer
"""

#%%
values = torch.tensor([
    [1.0, 0.9],
    [0.1, 0.2],
    [0.95, 0.85],
    [-0.8, -0.7]
])

print("\nValue Vectors:\n")
print(values)

#%%  [markdown]
"""
Values may differ from keys.

In transformers:

    values are learned representations
    carrying transferable information.
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 8 — WEIGHTED INFORMATION AGGREGATION
# ============================================================

Attention combines values using:

        attention weights

------------------------------------------------------------
# Mathematical Formulation
------------------------------------------------------------

Attention output:

        y = Σ(a_i v_i)

where:

    a_i = attention weight
    v_i = value vector

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Attention dynamically routes information.
"""

#%%
attention_output = torch.sum(
    attention_weights.unsqueeze(1)
    * values,
    dim=0
)

print("\nAttention Output:\n")
print(attention_output)

#%%  [markdown]
"""
Observe:

More relevant values contributed more strongly.

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Attention creates:

        adaptive information aggregation
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 9 — FULL ATTENTION PIPELINE
# ============================================================

Attention systems perform:

        relational interaction routing

------------------------------------------------------------
# Full Pipeline
------------------------------------------------------------

            Query
                ↓
        Relational Matching
                ↓
        Compatibility Scores
                ↓
              Softmax
                ↓
        Attention Weights
                ↓
        Weighted Aggregation
                ↓
         Contextual Output

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Attention dynamically determines:

        which information matters
"""

#%%
print("\nAttention Pipeline:\n")

print("Query")
print("   ↓")
print("Key Matching")
print("   ↓")
print("Compatibility Scores")
print("   ↓")
print("Softmax")
print("   ↓")
print("Attention Weights")
print("   ↓")
print("Weighted Values")
print("   ↓")
print("Contextual Representation")

#%%  [markdown]
"""
# ============================================================
# SECTION 10 — ATTENTION AS GEOMETRIC INTERACTION
# ============================================================

One of the deepest ideas in modern AI:

attention operates through:

        relational geometry

------------------------------------------------------------
# Important Concept
------------------------------------------------------------

Attention does NOT explicitly store rules like:

    "this word is important"

Instead:

    geometric compatibility determines
    interaction strength dynamically.

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Modern attention systems frequently operate through:

        similarity-driven geometry
"""

#%%
print("\nCore Attention Principle:\n")

print("Representations")
print("       ↓")
print("Geometric Compatibility")
print("       ↓")
print("Dynamic Interaction Weights")
print("       ↓")
print("Adaptive Information Flow")

#%%  [markdown]
"""
# ============================================================
# FINAL SUMMARY
# ============================================================

# Key Ideas Learned

1. Attention measures relational compatibility

2. Queries represent interaction intent

3. Keys represent searchable signatures

4. Dot products measure geometric alignment

5. Softmax converts compatibility into influence weights

6. Values carry transferable information

7. Attention dynamically routes information

------------------------------------------------------------
# Most Important Insight
------------------------------------------------------------

Attention systems dynamically determine:

        which representations
        should interact strongly

through:

        geometric similarity matching

inside representation space.

------------------------------------------------------------
# Connection to Future Topics
------------------------------------------------------------

This becomes foundational for:

    * self-attention
    * transformers
    * contextual representations
    * language models
    * multimodal interaction systems
"""