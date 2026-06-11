#%% [markdown]
"""
# ============================================================
# STEP 01 — Coordinates and Representation Spaces
# ============================================================

# Core Idea

Representations require coordinate systems.

Before studying:

    * embeddings
    * latent spaces
    * semantic geometry
    * transformers

we must first understand:

    coordinates encode relationships

------------------------------------------------------------
# Central Philosophy
------------------------------------------------------------

Modern AI systems rarely manipulate raw entities directly.

Instead they transform entities into:

    coordinates inside structured spaces

Examples:

    * GPS coordinates
    * RGB color coordinates
    * physical simulation states
    * sound frequencies
    * semantic embeddings

This means:

    representation becomes geometry

------------------------------------------------------------
# Mathematical Foundation
------------------------------------------------------------

A coordinate vector can be represented as:

                x = [x1, x2, x3, ..., xn]

where:

    xi = value along dimension i

A coordinate system creates a measurable space where:

    * distance
    * similarity
    * structure
    * neighborhoods

become mathematically observable.

------------------------------------------------------------
# Learning Goals
------------------------------------------------------------

By the end of this file you should understand:

    * coordinates as structured descriptions
    * representation spaces
    * geometric similarity
    * multidimensional organization
    * distance relationships
    * tensor-based representations

------------------------------------------------------------
"""

#%%
import torch

torch.manual_seed(42)

#%% [markdown]
"""
# ============================================================
# SECTION 1 — SIMPLE 2D COORDINATE SPACE
# ============================================================

A coordinate system defines positions.

In 2D space:

                    (x, y)

Each entity becomes a point.

------------------------------------------------------------
# Mathematical Representation
------------------------------------------------------------

A 2D point is:

                p = [x, y]

Example:

                p = [2, 5]

means:

    x-coordinate = 2
    y-coordinate = 5

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Coordinates are NOT merely numbers.

They define:

    structured positions
    inside measurable spaces
"""

#%%
points = torch.tensor([
    [1.0, 2.0],   # Point A
    [3.0, 1.0],   # Point B
    [6.0, 5.0],   # Point C
])

print("\n2D Coordinate Points:\n")
print(points)

print("\nTensor Shape:")
print(points.shape)

#%% [markdown]
"""
Interpretation:

                rows    → entities
                columns → coordinate dimensions

Tensor shape:

                (3, 2)

means:

    3 points
    2 coordinate dimensions
"""

#%% [markdown]
"""
# ============================================================
# SECTION 2 — DISTANCE AS RELATIONSHIP
# ============================================================

One of the most important ideas in representation systems:

                geometry encodes relationships

------------------------------------------------------------
# Euclidean Distance
------------------------------------------------------------

Distance between two points:

            d(a,b) = ||a - b||

Expanded form:

        d(a,b) = sqrt(
                        (x1 - x2)^2
                      + (y1 - y2)^2
                     )

------------------------------------------------------------
# Key Insight
------------------------------------------------------------

Closer points:
        more similar positions

Farther points:
        more different positions

This idea later becomes:

    * embedding similarity
    * semantic similarity
    * latent geometry
"""

#%%
point_a = points[0]
point_b = points[1]

distance_ab = torch.norm(point_a - point_b)

print("\nPoint A:")
print(point_a)

print("\nPoint B:")
print(point_b)

print("\nEuclidean Distance Between A and B:")
print(distance_ab)

#%% [markdown]
"""
Observe:

The subtraction:

            point_a - point_b

creates a displacement vector.

Norm operation:

            ||x||

measures displacement magnitude.

This converts geometry into measurable relationships.
"""

#%% [markdown]
"""
# ============================================================
# SECTION 3 — COORDINATES AS DESCRIPTIONS
# ============================================================

Coordinates can represent properties.

Suppose we describe animals using:

                [weight, speed]

This creates a representation space.

------------------------------------------------------------
# Representation Space
------------------------------------------------------------

A representation space is:

    a coordinate system where dimensions
    encode measurable properties

------------------------------------------------------------
# Important Concept
------------------------------------------------------------

Changing dimensions changes relationships.

Different coordinate systems expose:

    different structures
"""

#%%
animals = torch.tensor([
    [400.0, 60.0],   # Horse
    [4500.0, 25.0],  # Elephant
    [35.0, 80.0],    # Dog
    [0.2, 15.0],     # Mouse
])

animal_names = [
    "Horse",
    "Elephant",
    "Dog",
    "Mouse"
]

print("\nAnimal Representation Tensor:\n")
print(animals)

print("\nDimensions:")
print("[weight, speed]")

#%% [markdown]
"""
Interpretation:

Each row becomes:

            one entity representation

Example:

        [400, 60]

means:

    weight = 400
    speed  = 60

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

The animal itself is NOT stored.

Instead:

    measurable coordinates are stored.

Modern AI systems work similarly.
"""

#%% [markdown]
"""
# ============================================================
# SECTION 4 — RELATIONSHIP GEOMETRY
# ============================================================

Now we measure distances between representations.

------------------------------------------------------------
# Mathematical Interpretation
------------------------------------------------------------

Distance becomes:

        relational similarity

This is one of the deepest ideas in
modern representation learning.

------------------------------------------------------------
# Key Principle
------------------------------------------------------------

Similarity can emerge from geometry.
"""

#%%
print("\nPairwise Relationship Distances:\n")

for i in range(len(animals)):

    for j in range(i + 1, len(animals)):

        distance = torch.norm(
            animals[i] - animals[j]
        )

        print(
            f"{animal_names[i]} <-> "
            f"{animal_names[j]} : "
            f"{distance:.2f}"
        )

#%% [markdown]
"""
Observe:

Distances depend entirely on:

    the coordinate system

If dimensions change:

    relationships also change.

This idea becomes extremely important for:

    * embeddings
    * latent spaces
    * semantic systems
"""

#%% [markdown]
"""
# ============================================================
# SECTION 5 — HIGH DIMENSIONAL REPRESENTATIONS
# ============================================================

Real AI systems rarely use only 2 dimensions.

Modern systems may use:

    * 64 dimensions
    * 128 dimensions
    * 768 dimensions
    * thousands of dimensions

------------------------------------------------------------
# Mathematical Form
------------------------------------------------------------

An n-dimensional representation:

        x = [x1, x2, x3, ..., xn]

Each dimension contributes to:

    relational structure

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

High-dimensional spaces still preserve geometry.
"""

#%%
high_dim_vectors = torch.randn(4, 8)

print("\nHigh-Dimensional Representation Tensor:\n")
print(high_dim_vectors)

print("\nTensor Shape:")
print(high_dim_vectors.shape)

#%% [markdown]
"""
Tensor shape:

                (4, 8)

means:

    4 entities
    8-dimensional coordinate space
"""

#%% [markdown]
"""
# ============================================================
# SECTION 6 — GEOMETRY IN HIGH DIMENSIONS
# ============================================================

Even in high dimensions:

        distance relationships still exist

------------------------------------------------------------
# Important Concept
------------------------------------------------------------

Humans cannot visualize high-dimensional geometry directly.

But mathematics still operates consistently.

AI systems rely heavily on:

    high-dimensional geometry
"""

#%%
print("\nPairwise Distances in High-Dimensional Space:\n")

for i in range(len(high_dim_vectors)):

    for j in range(i + 1, len(high_dim_vectors)):

        distance = torch.norm(
            high_dim_vectors[i]
            - high_dim_vectors[j]
        )

        print(
            f"Entity {i} <-> Entity {j}"
            f" : {distance:.4f}"
        )

#%% [markdown]
"""
This is the beginning of:

    embedding geometry

Later:

    nearby vectors
        ↓
    similar meanings

becomes one of the foundations of modern AI.
"""

#%% [markdown]
"""
# ============================================================
# SECTION 7 — REPRESENTATION SPACES
# ============================================================

A representation space organizes entities into geometry.

------------------------------------------------------------
# Core Philosophy
------------------------------------------------------------

AI systems often manipulate:

        coordinates

instead of raw symbolic meaning.

------------------------------------------------------------
# Representation Flow
------------------------------------------------------------

                Entity
                    ↓
            Coordinate Vector
                    ↓
           Geometric Relationships
                    ↓
            Similarity Structure
                    ↓
          Representation System
"""

#%%
representation_vectors = torch.tensor([
    [0.1, 0.2, 0.9],
    [0.2, 0.1, 0.85],
    [0.9, 0.8, 0.2],
])

print("\nRepresentation Vectors:\n")
print(representation_vectors)

#%%
distance_01 = torch.norm(
    representation_vectors[0]
    - representation_vectors[1]
)

distance_02 = torch.norm(
    representation_vectors[0]
    - representation_vectors[2]
)

print("\nDistance Between Vector 0 and 1:")
print(distance_01)

print("\nDistance Between Vector 0 and 2:")
print(distance_02)

#%% [markdown]
"""
Observe:

Vector 0 and Vector 1 are closer.

This means:

    their representations are more similar.

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Meaningful structure can emerge from:

        coordinate organization

This idea later evolves into:

    * embeddings
    * latent spaces
    * transformers
    * vector databases
    * semantic retrieval
"""

#%% [markdown]
"""
# ============================================================
# SECTION 8 — REPRESENTATION GEOMETRY
# ============================================================

One of the deepest ideas in AI:

AI systems often do NOT understand objects directly.

Instead they manipulate:

        structured numerical geometry

------------------------------------------------------------
# Core Concept
------------------------------------------------------------

Modern representation systems operate through:

    coordinate transformations

not symbolic reasoning alone.

------------------------------------------------------------
# Extremely Important Insight
------------------------------------------------------------

Meaning can emerge from relational geometry.

This becomes one of the foundational ideas behind:

    * language models
    * recommendation systems
    * semantic search
    * multimodal AI
    * latent reasoning systems
"""

#%%
print("\nRepresentation Geometry Pipeline:\n")

print("Entity")
print("   ↓")
print("Coordinate Representation")
print("   ↓")
print("Geometric Relationships")
print("   ↓")
print("Similarity Structure")
print("   ↓")
print("Representation Space")

# %%[markdown]
"""
# ============================================================
# FINAL SUMMARY
# ============================================================

# Key Ideas Learned

1. Coordinates define measurable positions

2. Representation spaces organize relationships

3. Distance encodes similarity

4. Geometry exists even in high dimensions

5. AI systems manipulate representations
   inside coordinate spaces

6. Similarity structure can emerge from geometry

------------------------------------------------------------
# Most Important Insight
------------------------------------------------------------

Modern AI systems frequently transform:

            entities
                ↓
            coordinates
                ↓
            geometric relationships
                ↓
            structured representations

This becomes the foundation for:

    * embeddings
    * latent spaces
    * transformers
    * semantic systems
    * representation learning
"""