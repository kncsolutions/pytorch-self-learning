#%%  [markdown]
"""
# ============================================================
# STEP 01 — Attention as Dynamic Interaction
# ============================================================

# Core Idea

Attention dynamically selects important interactions.

One of the biggest conceptual shifts in modern AI:

not all representations should influence
each other equally.

Instead:

        interaction strength can change dynamically
depending on context and relational relevance.

------------------------------------------------------------
# Central Philosophy
------------------------------------------------------------

Attention is NOT magic focus.

Attention is:

        adaptive interaction weighting

inside representation systems.

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Traditional convolution systems often use:

        fixed local interactions

Attention systems introduce:

        dynamic interaction selection

------------------------------------------------------------
# Core Concept
------------------------------------------------------------

Some representations may be:

    * highly relevant
    * weakly relevant
    * irrelevant

depending on the current context.

------------------------------------------------------------
# Learning Goals
------------------------------------------------------------

By the end of this file you should understand:

    * dynamic interactions
    * interaction relevance
    * adaptive weighting
    * contextual influence
    * weighted aggregation
    * information routing
    * relational interaction systems

------------------------------------------------------------
"""

#%%
import torch

torch.manual_seed(42)

#%%  [markdown]
"""
# ============================================================
# SECTION 1 — FIXED INTERACTION SYSTEMS
# ============================================================

Traditional systems often use fixed interactions.

------------------------------------------------------------
# Example Intuition
------------------------------------------------------------

Suppose we have representations:

    x1, x2, x3, x4

A fixed interaction system may combine them equally.

------------------------------------------------------------
# Mathematical Formulation
------------------------------------------------------------

Simple aggregation:

        y = (x1 + x2 + x3 + x4) / 4

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

All representations contribute equally,
regardless of relevance.
"""

#%%
representations = torch.tensor([
    [0.9, 0.8],   # x1
    [0.1, 0.2],   # x2
    [0.85, 0.75], # x3
    [-0.9, -0.8]  # x4
])

print("\nRepresentations:\n")
print(representations)

fixed_output = representations.mean(dim=0)

print("\nFixed Aggregation Output:\n")
print(fixed_output)

#%%  [markdown]
"""
Observe:

Every representation influenced the output equally.

------------------------------------------------------------
# Important Limitation
------------------------------------------------------------

Real systems often require:

        selective interaction

instead of uniform interaction.
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 2 — DYNAMIC INTERACTION IDEA
# ============================================================

Attention introduces:

        adaptive interaction strength

------------------------------------------------------------
# Core Principle
------------------------------------------------------------

More relevant representations should contribute more.

Less relevant representations should contribute less.

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Attention creates:

        context-sensitive interaction systems
"""

#%%
attention_weights = torch.tensor([
    0.45,
    0.05,
    0.40,
    0.10
])

print("\nAttention Weights:\n")
print(attention_weights)

#%%  [markdown]
"""
Observe:

Some representations now receive:

    * stronger influence
    * weaker influence

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Attention redistributes interaction importance.
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 3 — WEIGHTED INTERACTION
# ============================================================

Attention performs weighted aggregation.

------------------------------------------------------------
# Mathematical Formulation
------------------------------------------------------------

Weighted interaction:

            y = Σ(w_i x_i)

where:

    w_i = interaction weight
    x_i = representation vector

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Attention controls:

        representational influence
"""

#%%
weighted_output = torch.sum(
    attention_weights.unsqueeze(1)
    * representations,
    dim=0
)

print("\nWeighted Attention Output:\n")
print(weighted_output)

#%%  [markdown]
"""
Observe:

Representations with larger weights contributed more.

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Attention changes:

        information flow geometry
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 4 — CONTEXTUAL RELEVANCE
# ============================================================

Attention weights depend on context.

------------------------------------------------------------
# Example Intuition
------------------------------------------------------------

Suppose we process the sentence:

    "The animal chased the cat."

Representations related to:

    * animal
    * chased
    * cat

may become more relevant together.

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Attention creates:

        context-dependent interaction
"""

#%%
tokens = [
    "animal",
    "tree",
    "cat",
    "river"
]

token_embeddings = torch.tensor([
    [0.9, 0.8],    # animal
    [-0.7, 0.1],   # tree
    [0.85, 0.75],  # cat
    [-0.8, -0.6]   # river
])

context_attention = torch.tensor([
    0.4,
    0.05,
    0.5,
    0.05
])

print("\nContextual Attention Weights:\n")

for token, weight in zip(tokens, context_attention):

    print(f"{token} : {weight}")

#%%  [markdown]
"""
Observe:

Contextually relevant tokens received:

        stronger interaction weights

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Attention dynamically adapts interaction structure.
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 5 — INFORMATION ROUTING
# ============================================================

Attention can be viewed as:

        information routing

------------------------------------------------------------
# Important Concept
------------------------------------------------------------

Attention determines:

    * which representations interact strongly
    * which information propagates
    * which signals dominate

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Attention systems dynamically control:

        information flow
"""

#%%
routed_output = torch.sum(
    context_attention.unsqueeze(1)
    * token_embeddings,
    dim=0
)

print("\nRouted Information Output:\n")
print(routed_output)

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

Attention performs:

        adaptive representational routing
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 6 — LOCAL VS DYNAMIC INTERACTIONS
# ============================================================

Traditional convolutions often rely on:

        fixed local neighborhoods

Attention systems can dynamically select:

        relevant interactions

------------------------------------------------------------
# Convolution Style
------------------------------------------------------------

Interaction depends mostly on:

        nearby positions

------------------------------------------------------------
# Attention Style
------------------------------------------------------------

Interaction depends on:

        relational relevance

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Attention generalizes interaction systems.
"""

#%%
sequence = torch.tensor([
    [0.9, 0.8],
    [0.1, 0.2],
    [0.85, 0.75],
    [-0.9, -0.8],
    [0.88, 0.82]
])

local_neighbors = sequence[:3]

local_output = local_neighbors.mean(dim=0)

dynamic_weights = torch.tensor([
    0.4,
    0.05,
    0.35,
    0.05,
    0.15
])

dynamic_output = torch.sum(
    dynamic_weights.unsqueeze(1)
    * sequence,
    dim=0
)

print("\nLocal Interaction Output:\n")
print(local_output)

print("\nDynamic Attention Output:\n")
print(dynamic_output)

#%%  [markdown]
"""
Observe:

Convolution-style interaction used:

        fixed neighborhood structure

Attention-style interaction used:

        adaptive weighting

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Attention enables:

        flexible relational interaction
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 7 — ATTENTION AS RELATIONAL FILTERING
# ============================================================

Attention can also be viewed as:

        relational filtering

------------------------------------------------------------
# Important Concept
------------------------------------------------------------

The system selectively amplifies:

    * useful signals
    * relevant representations

while suppressing:

    * irrelevant signals
    * noisy interactions

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Attention acts as:

        adaptive signal prioritization
"""

#%%
signals = torch.tensor([
    [1.0, 1.0],
    [0.1, 0.1],
    [0.9, 0.8],
    [-0.2, -0.1]
])

signal_weights = torch.tensor([
    0.45,
    0.05,
    0.45,
    0.05
])

filtered_output = torch.sum(
    signal_weights.unsqueeze(1)
    * signals,
    dim=0
)

print("\nFiltered Attention Output:\n")
print(filtered_output)

#%%  [markdown]
"""
Observe:

Weakly weighted signals contributed less.

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Attention performs:

        selective representational amplification
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 8 — ATTENTION IN MODERN AI
# ============================================================

Attention systems appear throughout modern AI.

------------------------------------------------------------
# Examples
------------------------------------------------------------

Attention is fundamental for:

    * transformers
    * language models
    * multimodal systems
    * recommendation systems
    * retrieval systems
    * vision transformers

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Modern AI frequently relies on:

        dynamic interaction systems
"""

#%%
applications = [
    "Transformers",
    "Large Language Models",
    "Vision Transformers",
    "Semantic Retrieval",
    "Recommendation Systems",
    "Multimodal AI"
]

print("\nApplications of Attention:\n")

for application in applications:
    print(f"- {application}")

#%%  [markdown]
"""
# ============================================================
# SECTION 9 — REPRESENTATION INTERACTION PIPELINE
# ============================================================

One of the deepest conceptual shifts in modern AI:

interaction strength can become dynamic.

------------------------------------------------------------
# Representation Flow
------------------------------------------------------------

            Representations
                    ↓
          Relational Relevance
                    ↓
            Attention Weights
                    ↓
         Weighted Interactions
                    ↓
          Contextual Output

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Attention systems dynamically determine:

        which representations matter most
"""

#%%
print("\nAttention Interaction Pipeline:\n")

print("Representations")
print("        ↓")
print("Relational Relevance")
print("        ↓")
print("Attention Weights")
print("        ↓")
print("Weighted Aggregation")
print("        ↓")
print("Contextual Representation")

#%%  [markdown]
"""
# ============================================================
# FINAL SUMMARY
# ============================================================

# Key Ideas Learned

1. Attention introduces dynamic interaction weighting

2. Not all representations contribute equally

3. Attention performs weighted aggregation

4. Attention adapts interaction structure contextually

5. Attention controls information flow

6. Attention generalizes fixed interaction systems

7. Modern AI systems frequently rely on adaptive interactions

------------------------------------------------------------
# Most Important Insight
------------------------------------------------------------

Attention systems dynamically determine:

        which representations
        should influence each other

inside representation geometry.

Attention becomes a mechanism for:

        adaptive interaction routing

rather than fixed local interaction.

------------------------------------------------------------
# Connection to Future Topics
------------------------------------------------------------

This becomes foundational for:

    * transformers
    * self-attention
    * contextual representations
    * semantic routing
    * language models
    * multimodal interaction systems
"""