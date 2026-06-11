#%%  [markdown]
"""
# ============================================================
# STEP 04 — Projection and Dimensionality
# ============================================================

# Core Idea

High-dimensional systems can be projected
into observable structure.

Modern AI systems frequently operate in:

    * 64 dimensions
    * 128 dimensions
    * 768 dimensions
    * thousands of dimensions

Humans cannot directly visualize such spaces.

Projection allows us to:

        observe structure indirectly.

------------------------------------------------------------
# Central Philosophy
------------------------------------------------------------

Projection is NOT the original structure.

Projection is:

        a lower-dimensional view
        of higher-dimensional geometry.

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Visualization is always an approximation.

Projected geometry may:

    * preserve some relationships
    * distort other relationships

------------------------------------------------------------
# Mathematical Formulation
------------------------------------------------------------

Suppose:

            x ∈ R^n

is an n-dimensional vector.

Projection transforms:

            x → y

where:

            y ∈ R^m

and:

            m < n

------------------------------------------------------------
# Learning Goals
------------------------------------------------------------

By the end of this file you should understand:

    * dimensionality
    * projection systems
    * dimensional reduction
    * information compression
    * projection distortion
    * observable geometry
    * low-dimensional views

------------------------------------------------------------
"""

#%%
import torch

torch.manual_seed(42)

#%%  [markdown]
"""
# ============================================================
# SECTION 1 — HIGH-DIMENSIONAL REPRESENTATIONS
# ============================================================

Real AI systems operate in high-dimensional spaces.

------------------------------------------------------------
# Mathematical Form
------------------------------------------------------------

A high-dimensional vector:

        x = [x1, x2, x3, ..., xn]

where:

    n may be very large.

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Humans struggle to visualize beyond:

    * 2D
    * 3D

But AI systems routinely operate in:

    hundreds or thousands of dimensions.
"""

#%%
high_dim_vectors = torch.randn(5, 8)

print("\nHigh-Dimensional Representation Tensor:\n")
print(high_dim_vectors)

print("\nTensor Shape:")
print(high_dim_vectors.shape)

#%%  [markdown]
"""
Tensor shape:

                (5, 8)

means:

    5 entities
    8-dimensional representation space

------------------------------------------------------------
# Important Clarification
------------------------------------------------------------

Even though we cannot visualize 8D space directly:

    geometry still exists mathematically.
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 2 — WHY PROJECTION IS NECESSARY
# ============================================================

Humans need lower-dimensional views.

Projection creates:

        observable approximations

of high-dimensional structure.

------------------------------------------------------------
# Projection Idea
------------------------------------------------------------

Suppose we have:

            x ∈ R^8

Projection may create:

            y ∈ R^2

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Projection compresses information.

Some structure survives.

Some structure is lost.
"""

#%%
print("\nOriginal 8D Vector:\n")
print(high_dim_vectors[0])

projected_vector = high_dim_vectors[0][:2]

print("\nProjected 2D Vector:\n")
print(projected_vector)

#%%  [markdown]
"""
Here we performed a very simple projection:

        first 8 dimensions
                ↓
            first 2 dimensions

------------------------------------------------------------
# Important Clarification
------------------------------------------------------------

Real dimensionality reduction methods are usually
more sophisticated than simple slicing.

But the central idea remains:

        high dimensions
                ↓
        lower-dimensional view
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 3 — PROJECTION AS INFORMATION COMPRESSION
# ============================================================

Projection compresses representation information.

------------------------------------------------------------
# Mathematical Interpretation
------------------------------------------------------------

Projection often reduces:

    * dimensional complexity
    * storage requirements
    * observable difficulty

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Compression usually introduces:

        information loss

This creates a tradeoff between:

    * simplicity
    * fidelity
"""

#%%
original_vector = torch.tensor([
    2.4,
    -1.1,
    0.9,
    4.2,
    -3.0,
    1.5
])

compressed_vector = original_vector[:2]

print("\nOriginal 6D Vector:\n")
print(original_vector)

print("\nCompressed 2D Projection:\n")
print(compressed_vector)

#%%  [markdown]
"""
Observe:

The projected vector contains only part
of the original information.

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Projection simplifies structure.

But simplification may distort relationships.
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 4 — DISTANCE DISTORTION
# ============================================================

Projection can change geometric relationships.

------------------------------------------------------------
# Important Concept
------------------------------------------------------------

Distances in projected space may differ from:

        original high-dimensional distances

------------------------------------------------------------
# Mathematical Idea
------------------------------------------------------------

Suppose:

        d_high(a,b)

is original distance.

After projection:

        d_low(a,b)

may change significantly.

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Projection may preserve:

    * local structure
    * global structure

but rarely preserves everything perfectly.
"""

#%%
vector_a = torch.tensor([
    1.0, 2.0, 3.0, 4.0
])

vector_b = torch.tensor([
    1.5, 2.5, 3.5, 10.0
])

high_dim_distance = torch.norm(
    vector_a - vector_b
)

projected_a = vector_a[:2]
projected_b = vector_b[:2]

low_dim_distance = torch.norm(
    projected_a - projected_b
)

print("\nOriginal High-Dimensional Distance:")
print(high_dim_distance)

print("\nProjected Low-Dimensional Distance:")
print(low_dim_distance)

#%%  [markdown]
"""
Observe:

Projected distance became much smaller.

Why?

Because projection removed dimensions containing
large differences.

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Projection may hide important structure.
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 5 — DIMENSIONALITY REDUCTION INTUITION
# ============================================================

Dimensionality reduction attempts to preserve:

        important structure

while reducing dimensional complexity.

------------------------------------------------------------
# Common Goals
------------------------------------------------------------

Dimensionality reduction may attempt to preserve:

    * distances
    * neighborhoods
    * clusters
    * variance
    * topology

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Different projection methods preserve
different kinds of structure.
"""

#%%
embeddings = torch.randn(10, 16)

print("\nOriginal Embedding Shape:")
print(embeddings.shape)

reduced_embeddings = embeddings[:, :2]

print("\nReduced Embedding Shape:")
print(reduced_embeddings.shape)

#%%  [markdown]
"""
Shape transition:

            (10, 16)
                    ↓
             (10, 2)

means:

    16-dimensional geometry
            ↓
       2D projection

------------------------------------------------------------
# Important Clarification
------------------------------------------------------------

This is a simplified educational example.

Real methods include:

    * PCA
    * t-SNE
    * UMAP
    * autoencoders
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 6 — PROJECTION MATRICES
# ============================================================

Projection can also be represented mathematically
using matrices.

------------------------------------------------------------
# Mathematical Formulation
------------------------------------------------------------

Suppose:

            x ∈ R^n

Projection matrix:

            P ∈ R^(m × n)

Projected vector:

                y = Px

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Projection becomes:

        structured geometric transformation
"""

#%%
vector = torch.tensor([
    3.0,
    2.0,
    1.0
])

projection_matrix = torch.tensor([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0]
])

projected = projection_matrix @ vector

print("\nOriginal Vector:\n")
print(vector)

print("\nProjection Matrix:\n")
print(projection_matrix)

print("\nProjected Vector:\n")
print(projected)

#%%  [markdown]
"""
Observe:

Matrix multiplication performed the projection.

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Projection is fundamentally:

        coordinate transformation

inside geometric space.
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 7 — LATENT STRUCTURE INTUITION
# ============================================================

Many high-dimensional systems contain hidden structure.

Projection sometimes reveals:

    * clusters
    * neighborhoods
    * manifolds
    * semantic organization

------------------------------------------------------------
# Important Idea
------------------------------------------------------------

High-dimensional representations are often NOT random.

They contain:

        structured organization

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Projection helps humans observe:

        hidden relational geometry
"""

#%%
cluster_1 = torch.randn(5, 4) + 5
cluster_2 = torch.randn(5, 4) - 5

combined = torch.cat([
    cluster_1,
    cluster_2
], dim=0)

projected_clusters = combined[:, :2]

print("\nProjected Cluster Coordinates:\n")
print(projected_clusters)

#%%  [markdown]
"""
Observe:

Even after projection:

    local grouping structure may survive.

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Projection can expose hidden organization
inside representation systems.
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 8 — REPRESENTATION GEOMETRY
# ============================================================

One of the deepest ideas in representation learning:

High-dimensional systems may contain:

        meaningful geometric organization

Projection gives humans partial access to
that hidden structure.

------------------------------------------------------------
# Representation Flow
------------------------------------------------------------

            Raw Data
                ↓
        High-Dimensional Representation
                ↓
        Projection / Compression
                ↓
         Observable Structure
                ↓
        Human Interpretation

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Projection does NOT create structure.

It reveals approximations of existing structure.
"""

#%%
print("\nProjection Pipeline:\n")

print("High-Dimensional Geometry")
print("            ↓")
print("Projection System")
print("            ↓")
print("Lower-Dimensional View")
print("            ↓")
print("Observable Structure")

#%%  [markdown]
"""
# ============================================================
# FINAL SUMMARY
# ============================================================

# Key Ideas Learned

1. AI systems operate in high-dimensional spaces

2. Projection creates lower-dimensional views

3. Projection compresses information

4. Dimensionality reduction simplifies geometry

5. Projection may distort relationships

6. Projection matrices perform coordinate transformation

7. Hidden structure may exist inside representation spaces

------------------------------------------------------------
# Most Important Insight
------------------------------------------------------------

Visualization is NOT the original structure.

Projection provides:

        partial observable views

of high-dimensional representation geometry.

------------------------------------------------------------
# Connection to Future Topics
------------------------------------------------------------

This becomes foundational for:

    * latent spaces
    * semantic geometry
    * manifold learning
    * autoencoders
    * transformers
    * representation learning
"""