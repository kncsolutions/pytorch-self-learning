#%%  [markdown]
"""
# ============================================================
# STEP 03 — Similarity and Distance Metrics
# ============================================================

# Core Idea

Similarity becomes measurable geometry.

One of the most important ideas in modern AI systems:

        relationships can be measured mathematically

inside representation spaces.

------------------------------------------------------------
# Central Philosophy
------------------------------------------------------------

Embeddings and representation vectors become useful
because we can measure:

    * similarity
    * distance
    * alignment
    * relational proximity

between vectors.

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Representation systems are fundamentally geometric systems.

Modern AI frequently operates by measuring:

        geometric relationships

between coordinate vectors.

------------------------------------------------------------
# Learning Goals
------------------------------------------------------------

By the end of this file you should understand:

    * distance metrics
    * similarity metrics
    * Euclidean geometry
    * cosine similarity
    * angular relationships
    * vector magnitude
    * relational geometry

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

Suppose we already have embeddings.

Each embedding becomes:

        a coordinate vector

inside representation space.

------------------------------------------------------------
# Mathematical Form
------------------------------------------------------------

Vector representation:

            x = [x1, x2, ..., xn]

where:

    xi = coordinate along dimension i

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Similarity depends on:

    * vector positions
    * vector directions
    * vector magnitudes
"""

#%%
cat_embedding = torch.tensor([
    0.9,
    0.7,
    0.2
])

dog_embedding = torch.tensor([
    0.85,
    0.75,
    0.25
])

car_embedding = torch.tensor([
    -0.8,
    -0.6,
    -0.3
])

print("\nCat Embedding:\n")
print(cat_embedding)

print("\nDog Embedding:\n")
print(dog_embedding)

print("\nCar Embedding:\n")
print(car_embedding)

#%%  [markdown]
"""
Observe:

Cat and Dog embeddings appear closer.

Car embedding appears more distant.

The next question becomes:

        how do we measure similarity?
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 2 — EUCLIDEAN DISTANCE
# ============================================================

Euclidean distance measures:

        geometric separation

between vectors.

------------------------------------------------------------
# Mathematical Formulation
------------------------------------------------------------

For vectors:

            a = [a1, a2, ..., an]
            b = [b1, b2, ..., bn]

Euclidean distance:

        d(a,b) = ||a - b||

Expanded form:

        d(a,b) = sqrt(
                            Σ(ai - bi)^2
                        )

------------------------------------------------------------
# Geometric Interpretation
------------------------------------------------------------

Small distance:
        nearby vectors

Large distance:
        distant vectors

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Distance creates measurable relationship geometry.
"""

#%%
cat_dog_distance = torch.norm(
    cat_embedding - dog_embedding
)

cat_car_distance = torch.norm(
    cat_embedding - car_embedding
)

print("\nEuclidean Distance — Cat ↔ Dog:")
print(cat_dog_distance)

print("\nEuclidean Distance — Cat ↔ Car:")
print(cat_car_distance)

#%%  [markdown]
"""
Observe:

Cat and Dog distance is smaller.

This suggests:

        stronger relational similarity

------------------------------------------------------------
# Important Clarification
------------------------------------------------------------

Distance does NOT automatically mean semantic truth.

Instead:

    geometry approximates relational structure.
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 3 — VECTOR MAGNITUDE
# ============================================================

Before cosine similarity, we must understand:

        vector magnitude

------------------------------------------------------------
# Mathematical Formulation
------------------------------------------------------------

Vector magnitude:

            ||x||

Expanded:

        ||x|| = sqrt(
                            Σ(xi)^2
                        )

------------------------------------------------------------
# Geometric Interpretation
------------------------------------------------------------

Magnitude measures:

        vector length

inside coordinate space.
"""

#%%
cat_magnitude = torch.norm(cat_embedding)

dog_magnitude = torch.norm(dog_embedding)

car_magnitude = torch.norm(car_embedding)

print("\nVector Magnitudes:\n")

print("Cat Magnitude :", cat_magnitude)
print("Dog Magnitude :", dog_magnitude)
print("Car Magnitude :", car_magnitude)

#%%  [markdown]
"""
Magnitude becomes important because:

two vectors may have:

    * different lengths
    * similar directions

This motivates cosine similarity.
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 4 — COSINE SIMILARITY
# ============================================================

Cosine similarity measures:

        directional alignment

between vectors.

------------------------------------------------------------
# Mathematical Formulation
------------------------------------------------------------
"""



#%%  [markdown]
"""
where:

    A · B     = dot product
    ||A||     = magnitude of A
    ||B||     = magnitude of B

------------------------------------------------------------
# Geometric Interpretation
------------------------------------------------------------

Cosine similarity measures:

        angular similarity

between vectors.

------------------------------------------------------------
# Value Interpretation
------------------------------------------------------------

Cosine similarity ranges between:

        -1  → opposite directions
         0  → orthogonal vectors
         1  → same direction

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Many embedding systems care more about:

        direction

than absolute distance.
"""

#%%
cosine_cat_dog = torch.nn.functional.cosine_similarity(
    cat_embedding,
    dog_embedding,
    dim=0
)

cosine_cat_car = torch.nn.functional.cosine_similarity(
    cat_embedding,
    car_embedding,
    dim=0
)

print("\nCosine Similarity — Cat ↔ Dog:")
print(cosine_cat_dog)

print("\nCosine Similarity — Cat ↔ Car:")
print(cosine_cat_car)

#%%  [markdown]
"""
Observe:

Cat and Dog vectors point in similar directions.

Cat and Car vectors point in opposite directions.

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Cosine similarity captures:

        relational orientation

inside representation space.
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 5 — DOT PRODUCT
# ============================================================

Cosine similarity depends on:

        dot product interaction

------------------------------------------------------------
# Mathematical Formulation
------------------------------------------------------------

Dot product:

        A · B = Σ(ai bi)

------------------------------------------------------------
# Geometric Interpretation
------------------------------------------------------------

The dot product measures:

    * directional agreement
    * interaction strength
    * vector alignment

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Many neural systems rely heavily on:

        vector interactions

through dot products.
"""

#%%
dot_cat_dog = torch.dot(
    cat_embedding,
    dog_embedding
)

dot_cat_car = torch.dot(
    cat_embedding,
    car_embedding
)

print("\nDot Product — Cat ↔ Dog:")
print(dot_cat_dog)

print("\nDot Product — Cat ↔ Car:")
print(dot_cat_car)

#%%  [markdown]
"""
Observe:

Positive dot product:
        aligned vectors

Negative dot product:
        opposing directions

This becomes foundational for:

    * transformers
    * attention systems
    * retrieval systems
    * semantic search
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 6 — DISTANCE VS SIMILARITY
# ============================================================

Distance and similarity are related,
but not identical.

------------------------------------------------------------
# Euclidean Distance
------------------------------------------------------------

Measures:

        absolute geometric separation

------------------------------------------------------------
# Cosine Similarity
------------------------------------------------------------

Measures:

        directional alignment

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Two vectors may:

    * point in similar directions
    * but have different magnitudes
"""

#%%
vector_a = torch.tensor([1.0, 1.0])

vector_b = torch.tensor([10.0, 10.0])

euclidean_distance = torch.norm(
    vector_a - vector_b
)

cosine_similarity = torch.nn.functional.cosine_similarity(
    vector_a,
    vector_b,
    dim=0
)

print("\nVector A:")
print(vector_a)

print("\nVector B:")
print(vector_b)

print("\nEuclidean Distance:")
print(euclidean_distance)

print("\nCosine Similarity:")
print(cosine_similarity)

#%%  [markdown]
"""
Observe:

Distance is large.

But cosine similarity is almost 1.

This means:

    vectors point in nearly the same direction.

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Direction and magnitude encode different information.
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 7 — SIMILARITY MATRICES
# ============================================================

Large AI systems often compare MANY vectors together.

------------------------------------------------------------
# Similarity Matrix
------------------------------------------------------------

A similarity matrix stores:

        pairwise relationships

between vectors.

------------------------------------------------------------
# Mathematical Form
------------------------------------------------------------

For vectors:

        x1, x2, ..., xn

similarity matrix:

                Sij = similarity(xi, xj)

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Representation systems often operate through:

        relational interaction maps
"""

#%%
embeddings = torch.tensor([
    [0.9, 0.7, 0.2],    # Cat
    [0.85, 0.75, 0.25], # Dog
    [-0.8, -0.6, -0.3], # Car
])

entity_names = [
    "Cat",
    "Dog",
    "Car"
]

print("\nPairwise Cosine Similarities:\n")

for i in range(len(embeddings)):

    for j in range(len(embeddings)):

        similarity = torch.nn.functional.cosine_similarity(
            embeddings[i],
            embeddings[j],
            dim=0
        )

        print(
            f"{entity_names[i]} ↔ "
            f"{entity_names[j]}"
            f" : {similarity:.4f}"
        )

#%%  [markdown]
"""
Observe:

Similarity matrices create:

        relational geometry maps

This becomes fundamental for:

    * attention systems
    * semantic retrieval
    * clustering
    * recommendation systems
"""

#%%  [markdown]
"""
# ============================================================
# SECTION 8 — REPRESENTATION GEOMETRY
# ============================================================

One of the deepest ideas in modern AI:

Representation systems frequently operate through:

        geometric interaction

------------------------------------------------------------
# Representation Flow
------------------------------------------------------------

                Entity
                    ↓
            Embedding Vector
                    ↓
            Geometric Position
                    ↓
          Similarity Measurement
                    ↓
          Relational Structure

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Modern AI systems often reason through:

        geometry

rather than symbolic rules alone.
"""

#%%
print("\nRepresentation Geometry Pipeline:\n")

print("Entity")
print("   ↓")
print("Embedding Vector")
print("   ↓")
print("Geometric Relationships")
print("   ↓")
print("Similarity Structure")
print("   ↓")
print("Representation System")

#%%  [markdown]
"""
# ============================================================
# FINAL SUMMARY
# ============================================================

# Key Ideas Learned

1. Similarity becomes measurable geometry

2. Euclidean distance measures separation

3. Vector magnitude measures length

4. Cosine similarity measures directional alignment

5. Dot products measure vector interaction

6. Distance and similarity capture different properties

7. Representation systems rely heavily on geometry

------------------------------------------------------------
# Most Important Insight
------------------------------------------------------------

Modern AI systems frequently compare:

        representation vectors

through geometric similarity metrics.

Meaningful structure can emerge from:

        relational geometry

inside high-dimensional spaces.

------------------------------------------------------------
# Connection to Future Topics
------------------------------------------------------------

This becomes foundational for:

    * embeddings
    * latent spaces
    * transformers
    * attention systems
    * vector databases
    * semantic retrieval
"""