#%%  [markdown]
"""
# ============================================================
# STEP 03 — Attention Weights and Information Flow
# ============================================================

# Core Idea

Attention redistributes representational influence.

One of the deepest conceptual ideas in attention systems:

attention determines:

        how information flows
between representations.

------------------------------------------------------------
# Central Philosophy
------------------------------------------------------------

Attention is NOT merely weighting.

It is:

        dynamic information routing

inside representation geometry.

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Representations with larger attention weights:

    * contribute more strongly
    * influence outputs more
    * dominate contextual aggregation

------------------------------------------------------------
# Information Flow Perspective
------------------------------------------------------------

Attention systems simulate:

        selective propagation of influence

through weighted interactions.

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

Attention weights determine:

        information flow intensity

------------------------------------------------------------
# Learning Goals
------------------------------------------------------------

By the end of this file you should understand:

    * attention weights
    * weighted information flow
    * contextual routing
    * influence propagation
    * signal amplification
    * information suppression
    * dynamic aggregation

------------------------------------------------------------
"""

#%%
import torch

torch.manual_seed(42)

#%%  [markdown]
"""
# ============================================================
# SECTION 1 — REPRESENTATION TOKENS
# ============================================================

Suppose we have multiple representations.

These may represent:

    * words
    * image patches
    * latent states
    * semantic embeddings

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Each representation contains:

        transferable information
"""

#%%
tokens = [
    "animal",
    "tree",
    "cat",
    "river"
]

value_vectors = torch.tensor([
    [0.9, 0.8],    # animal
    [0.1, 0.2],    # tree
    [0.85, 0.75],  # cat
    [-0.8, -0.7]   # river
])

print("\nValue Representations:\n")
print(value_vectors)

#%%  [markdown]
"""
Each row becomes:

        transferable representation information
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 2 — ATTENTION WEIGHTS
# ============================================================

Attention weights determine:

        interaction importance

------------------------------------------------------------
# Mathematical Interpretation
------------------------------------------------------------

Attention weights:

            a_i

control how strongly representation:

            v_i

influences the final output.

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Large attention weight:
        strong information flow

Small attention weight:
        weak information flow
"""

#%%
attention_weights = torch.tensor([
    0.45,  # animal
    0.05,  # tree
    0.40,  # cat
    0.10   # river
])

print("\nAttention Weights:\n")

for token, weight in zip(tokens, attention_weights):

    print(f"{token} : {weight}")

#%%  [markdown]
"""
Observe:

Animal and cat received stronger weights.

This means:

    their information will propagate more strongly.
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 3 — INFORMATION FLOW SIMULATION
# ============================================================

Attention simulates weighted information flow.

------------------------------------------------------------
# Mathematical Formulation
------------------------------------------------------------

Weighted contribution:

            flow_i = a_i * v_i

------------------------------------------------------------
# Geometric Interpretation
------------------------------------------------------------

Attention scales representation influence.

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Attention dynamically redistributes:

        representational importance
"""

#%%
weighted_flows = (
    attention_weights.unsqueeze(1)
    * value_vectors
)

print("\nWeighted Information Flows:\n")

for token, flow in zip(tokens, weighted_flows):

    print(f"{token} contribution:\n{flow}\n")

#%%  [markdown]
"""
Observe:

Representations with larger attention weights produced:

        stronger information flow vectors

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Attention acts like:

        adaptive signal amplification
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 4 — INFORMATION AGGREGATION
# ============================================================

Attention combines weighted flows into:

        contextual representations

------------------------------------------------------------
# Mathematical Formulation
------------------------------------------------------------

Contextual output:

            y = Σ(a_i v_i)

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

The final representation becomes:

        a weighted mixture
of relational information.
"""

#%%
contextual_output = weighted_flows.sum(dim=0)

print("\nContextual Representation:\n")
print(contextual_output)

#%%  [markdown]
"""
Observe:

The output became dominated by:

    * animal
    * cat

because they received stronger weights.

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Attention determines:

        which information survives strongly
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 5 — INFORMATION SUPPRESSION
# ============================================================

Attention can suppress weakly relevant signals.

------------------------------------------------------------
# Important Concept
------------------------------------------------------------

Small attention weights reduce:

        representational influence

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Attention performs:

        selective information filtering
"""

#%%
suppressed_weights = torch.tensor([
    0.48,
    0.01,
    0.50,
    0.01
])

suppressed_output = torch.sum(
    suppressed_weights.unsqueeze(1)
    * value_vectors,
    dim=0
)

print("\nSuppressed Attention Weights:\n")
print(suppressed_weights)

print("\nSuppressed Information Output:\n")
print(suppressed_output)

#%%  [markdown]
"""
Observe:

Tree and river contributed very little.

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Attention suppresses weakly relevant information.
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 6 — INFORMATION FLOW NETWORK
# ============================================================

Attention systems can be viewed as:

        dynamic interaction networks

------------------------------------------------------------
# Important Concept
------------------------------------------------------------

Representations exchange information
through weighted relational connections.

------------------------------------------------------------
# Information Flow Principle
------------------------------------------------------------

High compatibility:
        strong interaction pathway

Low compatibility:
        weak interaction pathway

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Attention dynamically reshapes:

        interaction topology
"""

#%%
interaction_matrix = torch.tensor([
    [0.50, 0.20, 0.25, 0.05],
    [0.10, 0.60, 0.20, 0.10],
    [0.35, 0.25, 0.30, 0.10],
    [0.05, 0.10, 0.15, 0.70]
])

print("\nAttention Interaction Matrix:\n")
print(interaction_matrix)

#%%  [markdown]
"""
Interpretation:

Row i:
        how representation i distributes attention

Column j:
        how strongly representation j receives influence

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Attention systems create:

        dynamic information routing graphs
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 7 — MULTI-REPRESENTATION FLOW SIMULATION
# ============================================================

Now we simulate full information propagation.

------------------------------------------------------------
# Mathematical Formulation
------------------------------------------------------------

Updated representations:

            Y = AV

where:

    A = attention matrix
    V = value matrix

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Attention transforms:

        representation interaction structure
"""

#%%
updated_representations = torch.matmul(
    interaction_matrix,
    value_vectors
)

print("\nUpdated Contextual Representations:\n")
print(updated_representations)

#%%  [markdown]
"""
Observe:

Each representation now contains:

        contextual information
from other representations.

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Attention enables:

        contextual information blending
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 8 — SIGNAL PROPAGATION INTUITION
# ============================================================

Attention systems behave like:

        adaptive signal propagation systems

------------------------------------------------------------
# Example Intuition
------------------------------------------------------------

Strongly weighted pathways transmit:

        more information

Weakly weighted pathways transmit:

        less information

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Attention controls:

        propagation strength
"""

#%%
signal_strengths = interaction_matrix.sum(dim=0)

print("\nTotal Incoming Signal Strengths:\n")

for token, strength in zip(tokens, signal_strengths):

    print(f"{token} : {strength:.4f}")

#%%  [markdown]
"""
Observe:

Some representations receive more total influence.

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Attention creates:

        unequal influence distribution
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 9 — ATTENTION AS DYNAMIC INFORMATION GEOMETRY
# ============================================================

One of the deepest ideas in transformers:

attention dynamically reshapes information geometry.

------------------------------------------------------------
# Representation Flow
------------------------------------------------------------

            Representations
                    ↓
          Relational Matching
                    ↓
           Attention Weights
                    ↓
          Weighted Information Flow
                    ↓
         Contextual Aggregation
                    ↓
        Updated Representations

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Attention systems continuously reorganize:

        representational influence
"""

#%%
print("\nAttention Information Flow Pipeline:\n")

print("Representations")
print("        ↓")
print("Relational Compatibility")
print("        ↓")
print("Attention Weights")
print("        ↓")
print("Weighted Information Flow")
print("        ↓")
print("Contextual Aggregation")
print("        ↓")
print("Updated Representations")

#%%  [markdown]
"""
# ============================================================
# SECTION 10 — INFORMATION FLOW IN MODERN AI
# ============================================================

Modern AI systems heavily rely on:

        dynamic information routing

------------------------------------------------------------
# Examples
------------------------------------------------------------

Attention-based information flow is fundamental for:

    * transformers
    * language models
    * vision transformers
    * multimodal systems
    * semantic retrieval
    * recommendation systems

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Modern AI frequently operates through:

        adaptive representational interaction
"""

#%%
applications = [
    "Transformers",
    "Large Language Models",
    "Vision Transformers",
    "Multimodal AI",
    "Semantic Retrieval",
    "Recommendation Systems"
]

print("\nApplications of Attention Information Flow:\n")

for application in applications:

    print(f"- {application}")

#%%  [markdown]
"""
# ============================================================
# FINAL SUMMARY
# ============================================================

# Key Ideas Learned

1. Attention weights determine information flow strength

2. Weighted aggregation creates contextual representations

3. Attention amplifies relevant information

4. Attention suppresses weakly relevant signals

5. Attention creates dynamic interaction networks

6. Attention continuously reshapes representational influence

7. Modern AI systems frequently rely on adaptive routing

------------------------------------------------------------
# Most Important Insight
------------------------------------------------------------

Attention systems dynamically determine:

        how information propagates

through representation geometry.

Attention becomes a mechanism for:

        adaptive information routing
        and contextual influence redistribution.

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