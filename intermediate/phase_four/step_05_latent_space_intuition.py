#%%  [markdown]
"""
# ============================================================
# STEP 05 — Latent Space Intuition
# ============================================================

# Core Idea

Latent spaces encode hidden relational structure.

Modern AI systems often transform raw data into:

        compressed internal representations

called:

        latent representations

------------------------------------------------------------
# What Does "Latent" Mean?
------------------------------------------------------------

Latent means:

        hidden
        underlying
        not directly observable

A latent space attempts to capture:

        important structure
        behind observable data.

------------------------------------------------------------
# Central Philosophy
------------------------------------------------------------

Latent spaces are NOT magical memory containers.

They are:

        compressed relational geometries

that organize patterns, behaviors,
or interaction structures.

------------------------------------------------------------
# Mathematical Formulation
------------------------------------------------------------

Suppose:

            x ∈ R^n

is original data.

A transformation function:

                z = f(x)

maps data into latent representation:

            z ∈ R^m

where:

            m < n

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Latent spaces attempt to preserve:

        meaningful structure

while compressing information.

------------------------------------------------------------
# Learning Goals
------------------------------------------------------------

By the end of this file you should understand:

    * latent representations
    * hidden structure
    * compressed geometry
    * representation compression
    * latent coordinates
    * relational organization
    * manifold intuition

------------------------------------------------------------
"""

#%%
import torch

torch.manual_seed(42)

#%%  [markdown]
"""
# ============================================================
# SECTION 1 — OBSERVABLE DATA VS LATENT STRUCTURE
# ============================================================

Raw data is often large and noisy.

Latent spaces attempt to capture:

        underlying organization

behind observable measurements.

------------------------------------------------------------
# Example Intuition
------------------------------------------------------------

Suppose we observe people through:

    * height
    * weight
    * age
    * heart rate
    * activity level
    * sleep patterns

But hidden beneath these measurements may exist:

    * fitness
    * stress
    * health state
    * lifestyle structure

These hidden variables resemble:

        latent structure

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Latent variables are often NOT directly observable.

They emerge indirectly from data relationships.
"""

#%%
observable_data = torch.tensor([
    [170.0, 65.0, 25.0, 70.0],
    [168.0, 63.0, 24.0, 72.0],
    [190.0, 95.0, 45.0, 88.0],
])

print("\nObservable Data:\n")
print(observable_data)

print("\nTensor Shape:")
print(observable_data.shape)

#%%  [markdown]
"""
Interpretation:

Columns may represent:

    [height, weight, age, heart_rate]

------------------------------------------------------------
# Important Clarification
------------------------------------------------------------

The system observes measurements.

But hidden structure may exist underneath.
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 2 — LATENT REPRESENTATIONS
# ============================================================

Latent representations compress information.

------------------------------------------------------------
# Mathematical Formulation
------------------------------------------------------------

Original representation:

            x ∈ R^n

Latent representation:

            z ∈ R^m

where:

            m < n

------------------------------------------------------------
# Compression Idea
------------------------------------------------------------

Latent spaces attempt to store:

        important relational structure

using fewer dimensions.

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Compression forces systems to discover:

        efficient structure
"""

#%%
latent_vectors = torch.tensor([
    [0.2, 0.8],
    [0.25, 0.75],
    [0.9, 0.1]
])

print("\nLatent Representations:\n")
print(latent_vectors)

print("\nTensor Shape:")
print(latent_vectors.shape)

#%%  [markdown]
"""
Shape transition:

            (3, 4)
                    ↓
             (3, 2)

means:

    original 4D observations
            ↓
    compressed 2D latent structure

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

The latent space attempts to preserve:

        relational organization

using fewer coordinates.
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 3 — LATENT GEOMETRY
# ============================================================

Latent spaces create geometric organization.

------------------------------------------------------------
# Important Idea
------------------------------------------------------------

Nearby latent vectors often represent:

    * similar behaviors
    * similar structures
    * similar interaction histories

------------------------------------------------------------
# Mathematical Relationship
------------------------------------------------------------

Distance:

            d(a,b) = ||a - b||

measures relational similarity.

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Latent geometry organizes hidden structure.
"""

#%%
distance_01 = torch.norm(
    latent_vectors[0]
    - latent_vectors[1]
)

distance_02 = torch.norm(
    latent_vectors[0]
    - latent_vectors[2]
)

print("\nDistance Between Latent Vector 0 and 1:")
print(distance_01)

print("\nDistance Between Latent Vector 0 and 2:")
print(distance_02)

#%%  [markdown]
"""
Observe:

Latent vectors 0 and 1 are closer.

This suggests:

        stronger hidden similarity

------------------------------------------------------------
# Important Clarification
------------------------------------------------------------

Latent spaces organize:

        relational patterns

not symbolic meaning directly.
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 4 — LATENT SPACES AS HIDDEN ORGANIZATION
# ============================================================

One of the deepest ideas in representation learning:

latent spaces encode hidden structure.

------------------------------------------------------------
# Example Intuition
------------------------------------------------------------

Suppose a music system observes:

    * rhythm
    * tempo
    * pitch
    * harmony

Latent structure may organize songs by:

    * mood
    * energy
    * genre
    * emotional similarity

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Latent spaces often capture:

        abstract relational organization
"""

#%%
music_latent = torch.tensor([
    [0.9, 0.8],   # energetic music
    [0.85, 0.75],
    [-0.7, -0.6], # calm music
    [-0.75, -0.55]
])

print("\nMusic Latent Space:\n")
print(music_latent)

#%%  [markdown]
"""
Observe:

Latent vectors form neighborhoods.

These neighborhoods may correspond to:

    shared hidden characteristics

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Latent spaces organize:

        similarity regions
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 5 — REPRESENTATION COMPRESSION
# ============================================================

Latent systems compress interaction history.

------------------------------------------------------------
# Compression Principle
------------------------------------------------------------

Suppose we have:

        large observable data

A latent system attempts to encode:

        essential structure

using fewer dimensions.

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Compression forces systems to prioritize:

        useful structure
"""

#%%
high_dim_data = torch.randn(5, 10)

compressed_data = high_dim_data[:, :3]

print("\nOriginal High-Dimensional Shape:")
print(high_dim_data.shape)

print("\nCompressed Latent Shape:")
print(compressed_data.shape)

#%%  [markdown]
"""
Shape transition:

            (5, 10)
                    ↓
             (5, 3)

means:

    high-dimensional observations
                ↓
       compressed latent coordinates

------------------------------------------------------------
# Important Clarification
------------------------------------------------------------

Real latent systems learn compression automatically.

This example only illustrates the intuition.
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 6 — LATENT NEIGHBORHOODS
# ============================================================

Latent spaces create neighborhoods.

------------------------------------------------------------
# Neighborhood Principle
------------------------------------------------------------

Nearby latent vectors often share:

    * behaviors
    * properties
    * interaction patterns
    * semantic structure

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Many AI systems operate through:

        neighborhood relationships
"""

#%%
query = latent_vectors[0]

print("\nQuery Latent Vector:\n")
print(query)

print("\nDistances From Query:\n")

for i in range(len(latent_vectors)):

    distance = torch.norm(
        query - latent_vectors[i]
    )

    print(f"Query ↔ Vector {i} : {distance:.4f}")

#%%  [markdown]
"""
Observe:

Nearest neighbors emerge geometrically.

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Latent spaces become:

        navigable relational systems
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 7 — LATENT SPACE TRAJECTORIES
# ============================================================

Latent spaces may contain smooth transitions.

------------------------------------------------------------
# Important Concept
------------------------------------------------------------

Small movement inside latent space may correspond to:

        gradual representation change

------------------------------------------------------------
# Example Intuition
------------------------------------------------------------

In image generation:

    latent movement may gradually change:

        * pose
        * lighting
        * style
        * expression

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Latent spaces often contain:

        continuous organization
"""

#%%
start_vector = torch.tensor([0.0, 0.0])

end_vector = torch.tensor([1.0, 1.0])

print("\nLatent Trajectory:\n")

for alpha in torch.linspace(0, 1, steps=5):

    interpolated = (
        (1 - alpha) * start_vector
        + alpha * end_vector
    )

    print(interpolated)

#%%  [markdown]
"""
Observe:

Intermediate vectors create smooth transitions.

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Latent spaces often support:

        continuous representation movement
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 8 — LATENT SPACES AND REPRESENTATION LEARNING
# ============================================================

One of the deepest ideas in AI:

systems often learn hidden relational structure
automatically from data.

------------------------------------------------------------
# Representation Flow
------------------------------------------------------------

            Raw Data
                ↓
        Representation Learning
                ↓
          Latent Coordinates
                ↓
         Geometric Structure
                ↓
        Relational Organization

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Latent spaces frequently organize:

        hidden interaction structure

inside compressed geometry.
"""

#%%
print("\nLatent Representation Pipeline:\n")

print("Raw Data")
print("    ↓")
print("Representation Learning")
print("    ↓")
print("Latent Coordinates")
print("    ↓")
print("Geometric Relationships")
print("    ↓")
print("Structured Representation")

#%%  [markdown]
"""
# ============================================================
# SECTION 9 — LATENT SPACES AND MODERN AI
# ============================================================

Latent spaces appear throughout modern AI systems.

------------------------------------------------------------
# Examples
------------------------------------------------------------

Latent representations are fundamental for:

    * embeddings
    * transformers
    * recommendation systems
    * generative AI
    * diffusion systems
    * semantic retrieval
    * multimodal learning

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Modern AI systems frequently operate by manipulating:

        compressed relational geometry
"""

#%%
print("\nExamples of Latent Systems:\n")

examples = [
    "Word Embeddings",
    "Image Latent Spaces",
    "Transformer Hidden States",
    "Recommendation Embeddings",
    "Diffusion Latent Representations",
]

for example in examples:
    print(f"- {example}")

#%%  [markdown]
"""
# ============================================================
# FINAL SUMMARY
# ============================================================

# Key Ideas Learned

1. Latent spaces encode hidden structure

2. Latent representations compress information

3. Latent geometry organizes relationships

4. Nearby latent vectors often share properties

5. Compression encourages efficient structure discovery

6. Latent spaces support smooth transitions

7. Modern AI frequently manipulates latent geometry

------------------------------------------------------------
# Most Important Insight
------------------------------------------------------------

Latent spaces are:

        compressed relational representation systems

where hidden structure becomes geometrically organized.

------------------------------------------------------------
# Connection to Future Topics
------------------------------------------------------------

This becomes foundational for:

    * semantic geometry
    * manifold learning
    * transformers
    * generative AI
    * diffusion models
    * multimodal systems
"""