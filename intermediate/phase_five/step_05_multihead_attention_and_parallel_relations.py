#%%  [markdown]
"""
# ============================================================
# STEP 05 — Multihead Attention and Parallel Relations
# ============================================================

# Core Idea

Different relational structures can be modeled simultaneously.

One of the deepest ideas in transformer systems:

a single interaction perspective is often insufficient.

Complex systems may require multiple parallel views of:

    * semantics
    * structure
    * syntax
    * temporal behavior
    * contextual relationships

------------------------------------------------------------
# Central Philosophy
------------------------------------------------------------

Multihead attention is NOT merely:

        "multiple attention blocks"

Instead it creates:

        parallel relational subspaces

inside representation geometry.

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Different attention heads may learn:

    * different interaction patterns
    * different contextual relationships
    * different representation structures

------------------------------------------------------------
# Core Idea
------------------------------------------------------------

Each attention head independently computes:

    * relational matching
    * attention weighting
    * contextual aggregation

------------------------------------------------------------
# Mathematical Formulation
------------------------------------------------------------
"""
#%%  [markdown]
"""
------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Each head creates:

        one relational perspective

------------------------------------------------------------
# Learning Goals
------------------------------------------------------------

By the end of this file you should understand:

    * multihead attention
    * parallel relational structure
    * contextual subspaces
    * multiple interaction perspectives
    * relational decomposition
    * contextual aggregation
    * representational specialization

------------------------------------------------------------
# Glossary of Important Functions
------------------------------------------------------------

torch.matmul(A, B)
    Matrix multiplication.

torch.softmax(x, dim)
    Converts scores into normalized attention weights.

torch.cat(tensors, dim)
    Concatenates tensors along a dimension.

transpose(dim0, dim1)
    Swaps tensor dimensions.

------------------------------------------------------------
# Glossary of Mathematical Concepts
------------------------------------------------------------

Attention Head:
        One independent relational interaction system.

Projection Matrix:
        Transforms representations into different subspaces.

Concatenation:
        Combines outputs from multiple heads.

Relational Subspace:
        A geometric region specializing in certain relationships.

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

Suppose we have representations for a sequence.

------------------------------------------------------------
# Mathematical Form
------------------------------------------------------------

            X ∈ R^(n × d)

where:

    n = number of representations
    d = representation dimension

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Multihead attention creates multiple relational
views of the same representations.
"""

#%%
tokens = [
    "animal",
    "hunts",
    "cat",
    "night"
]

X = torch.tensor([
    [0.9, 0.8, 0.2, 0.1],   # animal
    [0.7, 0.6, 0.8, 0.7],   # hunts
    [0.88, 0.82, 0.25, 0.15], # cat
    [0.2, 0.1, 0.95, 0.9]   # night
])

print("\nInput Representations:\n")
print(X)

print("\nTensor Shape:")
print(X.shape)

#%%  [markdown]
"""
Tensor shape:

            (4, 4)

means:

    4 representations
    4-dimensional embedding space
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 2 — MULTIPLE RELATIONAL HEADS
# ============================================================

Each attention head learns different relational structure.

------------------------------------------------------------
# Example Intuition
------------------------------------------------------------

Head 1 may focus on:

        semantic similarity

Head 2 may focus on:

        contextual behavior

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Different heads specialize in different patterns.
"""

#%%
# ------------------------------------------------------------
# HEAD 1 PROJECTIONS
# Semantic-style relationships
# ------------------------------------------------------------

W_Q1 = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [0.0, 0.0],
    [0.0, 0.0]
])

W_K1 = W_Q1.clone()
W_V1 = W_Q1.clone()

# ------------------------------------------------------------
# HEAD 2 PROJECTIONS
# Temporal/contextual-style relationships
# ------------------------------------------------------------

W_Q2 = torch.tensor([
    [0.0, 0.0],
    [0.0, 0.0],
    [1.0, 0.0],
    [0.0, 1.0]
])

W_K2 = W_Q2.clone()
W_V2 = W_Q2.clone()

#%%  [markdown]
"""
Observe:

Different projection matrices create:

        different relational subspaces

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Each head observes representations differently.
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 3 — HEAD 1: SEMANTIC RELATIONS
# ============================================================

Head 1 focuses mostly on:

        semantic similarity structure

------------------------------------------------------------
# Mathematical Formulation
------------------------------------------------------------

Q₁ = XW_Q₁
K₁ = XW_K₁
V₁ = XW_V₁
"""

#%%
Q1 = torch.matmul(X, W_Q1)
K1 = torch.matmul(X, W_K1)
V1 = torch.matmul(X, W_V1)

scores_1 = torch.matmul(
    Q1,
    K1.transpose(0, 1)
)

scaled_scores_1 = scores_1 / torch.sqrt(
    torch.tensor(float(K1.shape[1]))
)

attention_1 = torch.softmax(
    scaled_scores_1,
    dim=1
)

head_1_output = torch.matmul(
    attention_1,
    V1
)

print("\nHead 1 Attention Matrix:\n")
print(attention_1)

print("\nHead 1 Output:\n")
print(head_1_output)

#%%  [markdown]
"""
Observe:

Head 1 produced one style of relational interaction.

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

This head primarily captures:

        semantic geometric similarity
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 4 — HEAD 2: CONTEXTUAL RELATIONS
# ============================================================

Head 2 focuses mostly on:

        contextual / temporal structure

------------------------------------------------------------
# Mathematical Formulation
------------------------------------------------------------

Q₂ = XW_Q₂
K₂ = XW_K₂
V₂ = XW_V₂
"""

#%%
Q2 = torch.matmul(X, W_Q2)
K2 = torch.matmul(X, W_K2)
V2 = torch.matmul(X, W_V2)

scores_2 = torch.matmul(
    Q2,
    K2.transpose(0, 1)
)

scaled_scores_2 = scores_2 / torch.sqrt(
    torch.tensor(float(K2.shape[1]))
)

attention_2 = torch.softmax(
    scaled_scores_2,
    dim=1
)

head_2_output = torch.matmul(
    attention_2,
    V2
)

print("\nHead 2 Attention Matrix:\n")
print(attention_2)

print("\nHead 2 Output:\n")
print(head_2_output)

#%%  [markdown]
"""
Observe:

Head 2 produced a different interaction structure.

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Different heads capture:

        different relational geometry
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 5 — PARALLEL RELATIONAL STRUCTURES
# ============================================================

Each head independently computes:

    * relational matching
    * attention weighting
    * contextual aggregation

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Multihead attention enables:

        parallel relational reasoning
"""

#%%
print("\nComparing Head Outputs:\n")

print("Head 1 Output:\n")
print(head_1_output)

print("\nHead 2 Output:\n")
print(head_2_output)

#%%  [markdown]
"""
Observe:

The two heads produced:

        different contextual representations

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

The same input can generate:

        multiple relational interpretations
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 6 — CONCATENATING MULTIPLE HEADS
# ============================================================

Transformer systems combine head outputs.

------------------------------------------------------------
# Mathematical Formulation
------------------------------------------------------------

Combined output:

        H = Concat(head₁, head₂, ..., head_h)

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Concatenation merges multiple relational perspectives.
"""

#%%
multihead_output = torch.cat(
    [head_1_output, head_2_output],
    dim=1
)

print("\nCombined Multihead Output:\n")
print(multihead_output)

print("\nCombined Shape:")
print(multihead_output.shape)

#%%  [markdown]
"""
Observe:

The output dimension increased because:

        multiple relational views were combined.

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Multihead attention creates:

        richer contextual representations
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 7 — OUTPUT PROJECTION
# ============================================================

Transformer systems often apply:

        output projection

after concatenation.

------------------------------------------------------------
# Mathematical Formulation
------------------------------------------------------------

Final output:

            Y = HW_O

where:

    H   = concatenated heads
    W_O = output projection matrix

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Projection recombines relational information.
"""

#%%
W_O = torch.tensor([
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [0.0, 1.0]
])

final_output = torch.matmul(
    multihead_output,
    W_O
)

print("\nFinal Projected Output:\n")
print(final_output)

#%%  [markdown]
"""
Observe:

Multiple relational perspectives were merged
into one contextual representation.

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Multihead attention performs:

        relational information fusion
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 8 — INFORMATION FLOW COMPARISON
# ============================================================

Different heads create different attention pathways.

------------------------------------------------------------
# Important Concept
------------------------------------------------------------

Head specialization creates:

        parallel information routing systems

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Complex systems often require multiple
interaction perspectives simultaneously.
"""

#%%
print("\nHead 1 Information Flow:\n")
print(attention_1)

print("\nHead 2 Information Flow:\n")
print(attention_2)

#%%  [markdown]
"""
Observe:

The attention matrices differ significantly.

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Each head developed:

        distinct interaction topology
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 9 — MULTIHEAD ATTENTION IN MODERN AI
# ============================================================

Multihead attention became foundational for:

    * transformers
    * language models
    * vision transformers
    * multimodal AI
    * retrieval systems

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Modern AI systems frequently rely on:

        parallel relational subspaces
"""

#%%
applications = [
    "Transformers",
    "Large Language Models",
    "Vision Transformers",
    "Multimodal Systems",
    "Semantic Retrieval",
    "Generative AI"
]

print("\nApplications of Multihead Attention:\n")

for application in applications:

    print(f"- {application}")

#%%  [markdown]
"""
# ============================================================
# SECTION 10 — REPRESENTATION INTERACTION PIPELINE
# ============================================================

One of the deepest conceptual ideas in transformers:

different relational structures may coexist.

------------------------------------------------------------
# Representation Flow
------------------------------------------------------------

            Representations
                    ↓
         Multiple Projection Systems
                    ↓
          Parallel Attention Heads
                    ↓
        Distinct Relational Structures
                    ↓
            Contextual Aggregation
                    ↓
         Multihead Representation

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Multihead attention enables:

        parallel contextual interpretation
"""

#%%
print("\nMultihead Attention Pipeline:\n")

print("Representations")
print("       ↓")
print("Projection Subspaces")
print("       ↓")
print("Multiple Attention Heads")
print("       ↓")
print("Parallel Relational Geometry")
print("       ↓")
print("Contextual Aggregation")
print("       ↓")
print("Unified Representation")

#%%  [markdown]
"""
# ============================================================
# CONCLUSION FROM OUTPUTS
# ============================================================

# Observation 1

Head 1 and Head 2 produced:

        different attention matrices

This means:

        different relational structures emerged.

------------------------------------------------------------
# Observation 2

The same representations generated:

        multiple contextual interpretations

depending on projection subspaces.

------------------------------------------------------------
# Observation 3

Concatenating heads produced:

        richer representation geometry

than a single head alone.

------------------------------------------------------------
# Deepest Conceptual Insight
------------------------------------------------------------

Multihead attention allows systems to model:

        multiple relational perspectives
simultaneously.

Instead of one universal interaction structure,
the system constructs:

    * parallel contextual views
    * parallel information pathways
    * parallel relational geometries

------------------------------------------------------------
# Transformer-Level Insight
------------------------------------------------------------

This parallel relational processing became one of
the major breakthroughs behind transformers.

It enables systems to simultaneously model:

    * semantic structure
    * contextual structure
    * temporal relationships
    * symbolic interactions
    * latent dependencies

inside the same representation system.

------------------------------------------------------------
# FINAL SUMMARY
------------------------------------------------------------

# Key Ideas Learned

1. Multihead attention creates parallel relational subspaces

2. Different heads capture different interaction structures

3. Each head independently performs self-attention

4. Concatenation combines multiple contextual views

5. Output projection fuses relational information

6. Multihead systems produce richer representations

7. Modern transformers rely heavily on parallel interaction geometry

------------------------------------------------------------
# Most Important Insight
------------------------------------------------------------

Multihead attention enables representations to be interpreted through:

        multiple simultaneous relational perspectives

inside contextual representation geometry.
"""