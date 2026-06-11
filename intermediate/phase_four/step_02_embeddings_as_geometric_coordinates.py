#%% [markdown]
"""
# ============================================================
# STEP 02 — Embeddings as Geometric Coordinates
# ============================================================

# Core Idea

Embeddings place entities inside relational geometry.

An embedding transforms an entity into:

        a coordinate vector

inside a structured representation space.

------------------------------------------------------------
# Central Philosophy
------------------------------------------------------------

Embeddings are NOT magical containers of meaning.

They are:

        learned coordinate systems

where relationships become geometrically measurable.

------------------------------------------------------------
# Mathematical Formulation
------------------------------------------------------------

An embedding function:

                    f(x)

maps an entity:

                    x

into a vector space:

                f(x) ∈ R^n

where:

    R^n = n-dimensional coordinate space

------------------------------------------------------------
# Example
------------------------------------------------------------

A word:

                "cat"

may become:

        [0.12, -0.84, 0.33, ...]

This vector is NOT the meaning itself.

Instead:

    it represents a position
    inside relational geometry.

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Embeddings encode relationships through geometry.

Nearby embeddings:
        more similar relationships

Distant embeddings:
        more different relationships

------------------------------------------------------------
# Learning Goals
------------------------------------------------------------

By the end of this file you should understand:

    * embeddings as coordinates
    * learned representation spaces
    * geometric similarity
    * relational positioning
    * embedding tensors
    * local structure in vector spaces

------------------------------------------------------------
"""

#%%
import torch

torch.manual_seed(42)

#%% [markdown]
"""
# ============================================================
# SECTION 1 — EMBEDDINGS AS COORDINATE VECTORS
# ============================================================

An embedding is simply:

            a vector representation

of an entity.

------------------------------------------------------------
# Mathematical Representation
------------------------------------------------------------

Embedding vector:

            e = [e1, e2, e3, ..., en]

where:

    ei = coordinate value
         along embedding dimension i

------------------------------------------------------------
# Important Clarification
------------------------------------------------------------

Embedding dimensions usually do NOT have
direct human interpretation.

Instead:

    the full geometry matters.
"""

#%%
cat_embedding = torch.tensor([
    0.12,
    -0.84,
    0.33,
    0.91
])

dog_embedding = torch.tensor([
    0.15,
    -0.79,
    0.29,
    0.88
])

car_embedding = torch.tensor([
    -0.92,
    0.77,
    -0.44,
    -0.81
])

print("\nCat Embedding:\n")
print(cat_embedding)

print("\nDog Embedding:\n")
print(dog_embedding)

print("\nCar Embedding:\n")
print(car_embedding)

#%% [markdown]
"""
Observe:

Each entity becomes:

            coordinates in space

This creates:

        representation geometry

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

The embedding itself is NOT semantic meaning.

Instead:

    meaning emerges from
    relationships between embeddings.
"""

#%% [markdown]
"""
# ============================================================
# SECTION 2 — EMBEDDING SPACE GEOMETRY
# ============================================================

Embeddings create geometric organization.

------------------------------------------------------------
# Euclidean Distance
------------------------------------------------------------

Distance between embeddings:

                d(a,b) = ||a - b||

Expanded:

        d(a,b) = sqrt(
                            Σ(ai - bi)^2
                        )

------------------------------------------------------------
# Geometric Interpretation
------------------------------------------------------------

Small distance:
        nearby representations

Large distance:
        distant representations

------------------------------------------------------------
# Key Insight
------------------------------------------------------------

Similarity becomes measurable geometry.
"""

#%%
cat_dog_distance = torch.norm(
    cat_embedding - dog_embedding
)

cat_car_distance = torch.norm(
    cat_embedding - car_embedding
)

print("\nDistance Between Cat and Dog:")
print(cat_dog_distance)

print("\nDistance Between Cat and Car:")
print(cat_car_distance)

#%% [markdown]
"""
Observe:

Cat and Dog embeddings are closer.

This suggests:

    relational similarity

------------------------------------------------------------
# Important Clarification
------------------------------------------------------------

The system is NOT explicitly told:

        "cats and dogs are similar"

Instead:

    learning gradually organizes
    geometry relationally.
"""

#%% [markdown]
"""
# ============================================================
# SECTION 3 — EMBEDDING MATRICES
# ============================================================

Real AI systems store MANY embeddings together.

------------------------------------------------------------
# Mathematical Representation
------------------------------------------------------------

Embedding matrix:

                    E ∈ R^(N × D)

where:

    N = number of entities
    D = embedding dimension

------------------------------------------------------------
# Tensor Interpretation
------------------------------------------------------------

Rows:
        entity embeddings

Columns:
        embedding dimensions
"""

#%%
embedding_matrix = torch.tensor([
    [0.12, -0.84, 0.33, 0.91],   # Cat
    [0.15, -0.79, 0.29, 0.88],   # Dog
    [-0.92, 0.77, -0.44, -0.81], # Car
    [-0.88, 0.71, -0.39, -0.75], # Truck
])

entity_names = [
    "Cat",
    "Dog",
    "Car",
    "Truck"
]

print("\nEmbedding Matrix:\n")
print(embedding_matrix)

print("\nTensor Shape:")
print(embedding_matrix.shape)

#%% [markdown]
"""
Tensor shape:

                (4, 4)

means:

    4 entities
    4-dimensional embedding space

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Large AI systems may use:

    * 128 dimensions
    * 512 dimensions
    * 768 dimensions
    * thousands of dimensions

High-dimensional geometry becomes the foundation
of representation learning.
"""

#%% [markdown]
"""
# ============================================================
# SECTION 4 — RELATIONAL GEOMETRY
# ============================================================

Now we measure pairwise distances.

------------------------------------------------------------
# Core Philosophy
------------------------------------------------------------

Embeddings organize entities relationally.

This means:

    geometry stores structure.

------------------------------------------------------------
# Important Idea
------------------------------------------------------------

Embedding systems create:

        similarity neighborhoods
"""

#%%
print("\nPairwise Embedding Distances:\n")

for i in range(len(embedding_matrix)):

    for j in range(i + 1, len(embedding_matrix)):

        distance = torch.norm(
            embedding_matrix[i]
            - embedding_matrix[j]
        )

        print(
            f"{entity_names[i]} <-> "
            f"{entity_names[j]}"
            f" : {distance:.4f}"
        )

#%% [markdown]
"""
Observe:

    Cat ↔ Dog
    Car ↔ Truck

form local geometric neighborhoods.

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Embedding spaces naturally organize:

    related entities nearby

This becomes the basis for:

    * semantic search
    * recommendation systems
    * retrieval systems
    * language models
"""

#%% [markdown]
"""
# ============================================================
# SECTION 5 — EMBEDDINGS AS LEARNED COORDINATES
# ============================================================

One of the most important ideas in representation learning:

Embeddings are LEARNED.

------------------------------------------------------------
# Mathematical Idea
------------------------------------------------------------

Training gradually adjusts coordinates:

                E_new = E_old + ΔE

where:

    ΔE = learned coordinate update

------------------------------------------------------------
# Core Principle
------------------------------------------------------------

Learning reshapes geometry.

This creates:

    structured representation spaces.
"""

#%%
random_embeddings = torch.randn(5, 6)

print("\nRandom Initial Embeddings:\n")
print(random_embeddings)

#%% [markdown]
"""
Initially:

    embeddings may contain random geometry.

During training:

    geometry reorganizes itself.

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Learning often becomes:

        geometric restructuring

inside representation space.
"""

#%% [markdown]
"""
# ============================================================
# SECTION 6 — LOCAL STRUCTURE IN EMBEDDING SPACES
# ============================================================

Embedding systems create local neighborhoods.

------------------------------------------------------------
# Neighborhood Idea
------------------------------------------------------------

Nearby vectors often share:

    * properties
    * behaviors
    * semantic structure
    * interaction patterns

------------------------------------------------------------
# Important Concept
------------------------------------------------------------

Local geometry becomes meaningful structure.
"""

#%%
query_embedding = cat_embedding

print("\nQuery Embedding (Cat):\n")
print(query_embedding)

print("\nDistances From Query:\n")

for i, entity in enumerate(entity_names):

    distance = torch.norm(
        query_embedding
        - embedding_matrix[i]
    )

    print(f"Cat <-> {entity} : {distance:.4f}")

#%% [markdown]
"""
Observe:

Closest entities become nearest neighbors.

This idea becomes fundamental for:

    * vector databases
    * retrieval systems
    * recommendation engines
    * semantic search

------------------------------------------------------------
# Important Insight
------------------------------------------------------------

Many AI systems operate through:

        nearest-neighbor geometry
"""

#%% [markdown]
"""
# ============================================================
# SECTION 7 — EMBEDDINGS AS REPRESENTATION SYSTEMS
# ============================================================

Embeddings are more than vectors.

They create:

        navigable relational spaces

------------------------------------------------------------
# Representation Pipeline
------------------------------------------------------------

                Entity
                    ↓
            Embedding Function
                    ↓
            Coordinate Vector
                    ↓
           Geometric Structure
                    ↓
          Relational Organization

------------------------------------------------------------
# Extremely Important Insight
------------------------------------------------------------

Modern AI systems frequently operate by:

        transforming geometry

rather than manipulating symbolic meaning directly.
"""

#%%
print("\nEmbedding Representation Pipeline:\n")

print("Entity")
print("   ↓")
print("Embedding Function")
print("   ↓")
print("Coordinate Vector")
print("   ↓")
print("Geometric Relationships")
print("   ↓")
print("Representation Structure")

#%% [markdown]
"""
# ============================================================
# FINAL SUMMARY
# ============================================================

# Key Ideas Learned

1. Embeddings are coordinate vectors

2. Embedding spaces organize relationships

3. Similarity becomes measurable geometry

4. Embedding matrices store many representations

5. Learning reshapes geometric structure

6. Local neighborhoods encode relational similarity

7. Modern AI systems frequently manipulate geometry

------------------------------------------------------------
# Most Important Insight
------------------------------------------------------------

Embeddings transform entities into:

            navigable geometric coordinates

inside representation spaces.

Meaning emerges from:

        relational positioning

rather than isolated symbolic definitions.

------------------------------------------------------------
# Connection to Future Topics
------------------------------------------------------------

This becomes the foundation for:

    * latent spaces
    * transformers
    * semantic search
    * vector databases
    * representation learning
    * multimodal systems
"""