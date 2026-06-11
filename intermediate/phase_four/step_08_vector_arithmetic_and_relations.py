#%%  [markdown]
"""
# ============================================================
# STEP 08 — Vector Arithmetic and Relations
# ============================================================

# Core Idea

Vector displacement can encode relational transformations.

One of the most fascinating discoveries in
representation learning:

        relationships themselves
can become geometric directions.

------------------------------------------------------------
# Central Philosophy
------------------------------------------------------------

Embeddings are not merely stored coordinates.

They also contain:

        relational structure

inside geometric space.

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

A displacement vector may represent:

    * semantic change
    * relational transformation
    * contextual transition
    * structural movement

------------------------------------------------------------
# Mathematical Formulation
------------------------------------------------------------

Suppose:

        x_a, x_b ∈ R^n

Then:

        r = x_b - x_a

creates a relation vector.

------------------------------------------------------------
# Example
------------------------------------------------------------

If:

        king → queen

represents a gender transformation,

then similar relational movement may appear in:

        man → woman

------------------------------------------------------------
# Important Clarification
------------------------------------------------------------

Vector arithmetic does NOT mean the system
symbolically understands concepts.

Instead:

    relational geometry may become structured
    through learning dynamics.

------------------------------------------------------------
# Learning Goals
------------------------------------------------------------

By the end of this file you should understand:

    * vector displacement
    * relational transformations
    * vector arithmetic
    * semantic directions
    * geometric analogy structure
    * translation in latent space
    * relational geometry

------------------------------------------------------------
"""

#%%
import torch

torch.manual_seed(42)

#%%  [markdown]
"""
# ============================================================
# SECTION 1 — VECTOR DISPLACEMENT
# ============================================================

A displacement vector measures:

        movement inside representation space.

------------------------------------------------------------
# Mathematical Formulation
------------------------------------------------------------

Given:

        a = [a1, a2, ..., an]
        b = [b1, b2, ..., bn]

Displacement:

            r = b - a

------------------------------------------------------------
# Geometric Interpretation
------------------------------------------------------------

The displacement vector represents:

        direction + magnitude

of transformation.

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Relations themselves may become geometric objects.
"""

#%%
man = torch.tensor([
    0.2,
    0.4,
    0.6
])

woman = torch.tensor([
    0.3,
    0.5,
    0.7
])

relation_vector = woman - man

print("\nMan Vector:\n")
print(man)

print("\nWoman Vector:\n")
print(woman)

print("\nRelation Vector (woman - man):\n")
print(relation_vector)

#%%  [markdown]
"""
Observe:

Subtraction produced a transformation vector.

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Vector subtraction converts:

        positional difference
                ↓
        relational direction
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 2 — RELATIONAL TRANSFORMATIONS
# ============================================================

Suppose relational movement is consistent.

Then similar transformations may appear elsewhere.

------------------------------------------------------------
# Example Intuition
------------------------------------------------------------

If:

        man → woman

represents a relational shift,

then applying similar movement to:

        king

may approach:

        queen

------------------------------------------------------------
# Mathematical Formulation
------------------------------------------------------------

Given:

        r = woman - man

Apply transformation:

        queen ≈ king + r

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Relations may become reusable geometric directions.
"""

#%%
king = torch.tensor([
    0.8,
    0.9,
    0.7
])

predicted_queen = king + relation_vector

print("\nKing Vector:\n")
print(king)

print("\nPredicted Queen Vector:\n")
print(predicted_queen)

#%%  [markdown]
"""
Observe:

The relation vector shifted the representation.

------------------------------------------------------------
# Important Clarification
------------------------------------------------------------

This is NOT symbolic reasoning.

Instead:

    representation geometry contains
    structured directional patterns.
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 3 — VECTOR ARITHMETIC
# ============================================================

Vector arithmetic manipulates relationships geometrically.

------------------------------------------------------------
# Mathematical Operations
------------------------------------------------------------

Addition:

            a + b

Subtraction:

            a - b

Scaling:

            αx

------------------------------------------------------------
# Geometric Interpretation
------------------------------------------------------------

Addition:
        movement through space

Subtraction:
        relational displacement

Scaling:
        directional stretching

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Representation systems frequently operate through:

        geometric transformation
"""

#%%
vector_a = torch.tensor([
    1.0,
    2.0
])

vector_b = torch.tensor([
    0.5,
    1.5
])

print("\nVector A:\n")
print(vector_a)

print("\nVector B:\n")
print(vector_b)

print("\nA + B:\n")
print(vector_a + vector_b)

print("\nA - B:\n")
print(vector_a - vector_b)

print("\n2 * A:\n")
print(2 * vector_a)

#%%  [markdown]
"""
Observe:

Vector arithmetic creates geometric movement.

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Representation manipulation often becomes:

        geometric navigation
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 4 — RELATIONAL DIRECTIONS
# ============================================================

Embedding spaces may contain:

        meaningful directions

------------------------------------------------------------
# Example Intuition
------------------------------------------------------------

Possible relational directions:

    * gender shift
    * tense transformation
    * emotional intensity
    * object size
    * semantic category

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Directions may encode:

        structured relational change
"""

#%%
small = torch.tensor([
    0.2,
    0.1
])

large = torch.tensor([
    0.8,
    0.7
])

size_direction = large - small

print("\nSmall Vector:\n")
print(small)

print("\nLarge Vector:\n")
print(large)

print("\nSize Direction:\n")
print(size_direction)

#%%  [markdown]
"""
Observe:

The direction vector represents:

        movement toward larger representation values.

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Relational structure may emerge as:

        directional geometry
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 5 — GEOMETRIC ANALOGIES
# ============================================================

Some embedding systems exhibit analogy structure.

------------------------------------------------------------
# Example
------------------------------------------------------------

Classic embedding relation:

        king - man + woman ≈ queen

------------------------------------------------------------
# Mathematical Interpretation
------------------------------------------------------------

The system attempts to preserve:

        relational consistency

inside geometric space.

------------------------------------------------------------
# Important Clarification
------------------------------------------------------------

This does NOT imply symbolic understanding.

Instead:

    similar interaction structures may produce
    similar geometric displacements.
"""

#%%
king = torch.tensor([
    0.9,
    0.8,
    0.7
])

man = torch.tensor([
    0.2,
    0.3,
    0.4
])

woman = torch.tensor([
    0.3,
    0.4,
    0.5
])

queen_estimate = king - man + woman

print("\nKing Vector:\n")
print(king)

print("\nQueen Estimate:\n")
print(queen_estimate)

#%%  [markdown]
"""
Observe:

Arithmetic creates relational transformation.

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Embedding spaces may preserve:

        transformation consistency
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 6 — LATENT SPACE TRANSLATION
# ============================================================

Vector arithmetic creates movement through latent space.

------------------------------------------------------------
# Important Concept
------------------------------------------------------------

Adding displacement vectors performs:

        representation translation

------------------------------------------------------------
# Geometric Interpretation
------------------------------------------------------------

A vector shift changes:

        latent position

inside representation geometry.

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Latent spaces support:

        continuous transformation
"""

#%%
start = torch.tensor([
    0.0,
    0.0
])

direction = torch.tensor([
    1.0,
    0.5
])

print("\nLatent Space Translation:\n")

for alpha in torch.linspace(0, 1, steps=5):

    translated = start + alpha * direction

    print(translated)

#%%  [markdown]
"""
Observe:

Gradual translation creates smooth movement.

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Latent geometry often supports:

        continuous relational trajectories
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 7 — RELATIONAL STRUCTURE IN MODERN AI
# ============================================================

Vector arithmetic appears throughout modern AI systems.

------------------------------------------------------------
# Examples
------------------------------------------------------------

Relational vector geometry is important for:

    * word embeddings
    * transformers
    * recommendation systems
    * semantic retrieval
    * multimodal learning
    * latent generative systems

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Modern AI systems frequently manipulate:

        relational directions

inside representation spaces.
"""

#%%
applications = [
    "Word Embeddings",
    "Transformer Representations",
    "Semantic Retrieval",
    "Recommendation Embeddings",
    "Latent Generative Models"
]

print("\nApplications of Vector Relations:\n")

for application in applications:
    print(f"- {application}")

#%%  [markdown]
"""
# ============================================================
# SECTION 8 — REPRESENTATION GEOMETRY PIPELINE
# ============================================================

One of the deepest ideas in representation learning:

relations themselves may become geometric structure.

------------------------------------------------------------
# Representation Flow
------------------------------------------------------------

            Interaction Patterns
                    ↓
          Representation Learning
                    ↓
            Embedding Geometry
                    ↓
          Relational Directions
                    ↓
         Vector Transformations

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Modern AI systems frequently manipulate:

        relational geometry

through vector operations.
"""

#%%
print("\nVector Relation Pipeline:\n")

print("Interaction Patterns")
print("         ↓")
print("Representation Learning")
print("         ↓")
print("Embedding Coordinates")
print("         ↓")
print("Relational Directions")
print("         ↓")
print("Geometric Transformations")

#%%  [markdown]
"""
# ============================================================
# FINAL SUMMARY
# ============================================================

# Key Ideas Learned

1. Vector subtraction creates relational displacement

2. Relations may become geometric directions

3. Vector arithmetic enables representation transformation

4. Embedding spaces may preserve analogy structure

5. Latent spaces support continuous translation

6. Relational consistency may emerge geometrically

7. Modern AI systems frequently manipulate vector geometry

------------------------------------------------------------
# Most Important Insight
------------------------------------------------------------

Representation systems may organize relationships as:

        geometric transformations

inside latent space.

Relational structure can emerge from:

        directional geometry
        in representation systems.

------------------------------------------------------------
# Connection to Future Topics
------------------------------------------------------------

This becomes foundational for:

    * transformers
    * attention systems
    * semantic retrieval
    * latent generative models
    * multimodal AI
    * representation reasoning
"""