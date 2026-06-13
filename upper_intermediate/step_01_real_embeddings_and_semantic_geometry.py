#%% [markdown]
# ============================================================
# PHASE 06 — APPLIED REPRESENTATION ANALYSIS
# STEP 01 — REAL EMBEDDINGS AND SEMANTIC GEOMETRY
# ============================================================

# Repository Philosophy:
#
# Tensor intuition first.
# Representation systems second.
# Architecture complexity later.
#
# ------------------------------------------------------------
# CENTRAL QUESTION OF THIS STEP
# ------------------------------------------------------------
#
# How does semantic meaning become geometric structure?
#
# Modern AI systems do not store meaning symbolically like:
#
#     "cat" -> predefined symbolic definition
#
# Instead:
#
# they transform concepts into numerical vectors
# inside high-dimensional geometric spaces.
#
# In this step we investigate:
#
#     • semantic similarity
#     • geometric proximity
#     • representation organization
#     • embedding neighborhoods
#     • latent semantic structure
#
# using REAL pretrained embedding systems.
#
# ------------------------------------------------------------
# WHAT MAKES THIS STEP IMPORTANT?
# ------------------------------------------------------------
#
# Embeddings are foundational to:
#
#     • LLMs
#     • transformers
#     • semantic search
#     • recommendation systems
#     • retrieval systems
#     • multimodal learning
#     • RAG pipelines
#     • vector databases
#
# Without embeddings:
#
# modern AI systems would not possess
# semantic organization mechanisms.
#
# ============================================================


#%% [markdown]

#==========================================================
# 1. CONCEPTUAL INTRODUCTION
# ============================================================

"""
CONCEPTUAL FOUNDATION
---------------------

Traditional symbolic systems operate using discrete symbols.

Example:

    cat != dog

No geometric relationship exists.

But embedding systems transform symbols into vectors:

    cat  -> [0.12, -0.55, ..., 0.91]
    dog  -> [0.09, -0.49, ..., 0.88]

Now concepts can occupy positions inside a geometric space.

This introduces:

    • distance
    • neighborhoods
    • direction
    • clustering
    • semantic similarity

QUESTION:
---------

What if meaning itself could emerge
through geometric organization?

This is one of the deepest ideas in modern AI.

------------------------------------------------------------
EMBEDDING INTUITION
------------------------------------------------------------

Imagine a giant semantic universe.

Concepts with similar meanings
occupy nearby regions.

Example:

    king ---- queen
      |
    prince -- princess

Animals may form one region.

Emotions another.

Politics another.

Science another.

The embedding model learns these structures
automatically from data.

------------------------------------------------------------
IMPORTANT TRANSITION
------------------------------------------------------------

We are no longer studying:

    tensors as containers

We are now studying:

    tensors as semantic states

This is a major conceptual shift.
"""

#%% [markdown]

#==========================================================
# 2. REPRESENTATION HYPOTHESIS
# ============================================================

"""
REPRESENTATION HYPOTHESIS
-------------------------

Hypothesis:

    Semantically related words
    should occupy nearby regions
    in embedding space.

Expected observations:

    • cat should be near dog
    • king should be near queen
    • apple may appear near fruit
    • doctor may appear near hospital

Why?

Because embedding systems learn statistical
co-occurrence structure from language.

------------------------------------------------------------
SCIENTIFIC QUESTION
------------------------------------------------------------

If two concepts appear in similar contexts,
will their geometric representations become similar?

This step investigates that question directly.

------------------------------------------------------------
THINK DEEPLY
------------------------------------------------------------

What does "nearby" mean mathematically?

Nearby in:

    • Euclidean distance?
    • cosine similarity?
    • angular similarity?
    • latent topology?

What kind of geometry does language create?
"""



#%% ==========================================================
# 3. LOAD REAL EMBEDDING SYSTEM
# ============================================================

"""
We now load a REAL pretrained embedding system.

We intentionally avoid toy embeddings.

We want learners to observe
real semantic organization.

------------------------------------------------------------
MODEL CHOICE
------------------------------------------------------------

We use sentence-transformers because:

    • high-quality pretrained embeddings
    • industry relevance
    • transformer-based representations
    • easy semantic analysis

Model:

    all-MiniLM-L6-v2

This model maps text into
384-dimensional semantic vectors.

------------------------------------------------------------
GLOSSARY OF FUNCTIONS
------------------------------------------------------------

SentenceTransformer(...)
    Loads pretrained embedding model.

model.encode(...)
    Converts text into embedding vectors.

convert_to_tensor=True
    Returns PyTorch tensors instead of NumPy arrays.

------------------------------------------------------------
IMPORTANT INSIGHT
------------------------------------------------------------

The model was NOT explicitly taught:

    "cat is semantically similar to dog"

Instead:

semantic geometry emerged from training.
"""



#%%

from sentence_transformers import SentenceTransformer

# Load pretrained embedding system
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding system loaded successfully.")


#%% [markdown]
#==========================================================
# 4. EXTRACT EMBEDDINGS
# ============================================================

"""
We now convert language into geometry.

------------------------------------------------------------
QUESTION
------------------------------------------------------------

What happens when language
is transformed into vectors?

Does meaning survive the transformation?

------------------------------------------------------------
WORDS / CONCEPTS TO ANALYZE
------------------------------------------------------------

We intentionally choose concepts
from different semantic regions.

Observe:

    • animals
    • royalty
    • emotions
    • science
    • technology

------------------------------------------------------------
IMPORTANT IDEA
------------------------------------------------------------

Each embedding is NOT just data.

Each embedding is:

    a coordinate inside semantic space.
"""



#%%

concepts = [
    "cat",
    "dog",
    "lion",
    "tiger",
    "king",
    "queen",
    "prince",
    "princess",
    "computer",
    "software",
    "physics",
    "chemistry",
    "happy",
    "sad",
    "angry",
]

# Extract embeddings
embeddings = model.encode(
    concepts,
    convert_to_tensor=True
)

print("Embedding tensor shape:", embeddings.shape)


#%% [markdown]
#%% ==========================================================
# 5. SEMANTIC SIMILARITY ANALYSIS
# ============================================================

"""
Semantic similarity becomes measurable geometry.

------------------------------------------------------------
CORE IDEA
------------------------------------------------------------

If two vectors point in similar directions,
their concepts may possess related meaning.

We use:

    cosine similarity

------------------------------------------------------------
WHY COSINE SIMILARITY?
------------------------------------------------------------

Cosine similarity measures angular similarity.

Instead of asking:

    "How far apart are vectors?"

we ask:

    "Do vectors point in similar directions?"

This is often more meaningful
in high-dimensional spaces.

------------------------------------------------------------
COSINE SIMILARITY
------------------------------------------------------------

Values:

     1.0  -> very similar
     0.0  -> unrelated
    -1.0  -> opposite direction

------------------------------------------------------------
THINK CAREFULLY
------------------------------------------------------------

Why might angle matter more than distance
inside high-dimensional representation spaces?
"""



#%%

import torch
from sentence_transformers.util import cos_sim

# Compute pairwise cosine similarity matrix
similarity_matrix = cos_sim(embeddings, embeddings)

print("Similarity matrix shape:")
print(similarity_matrix.shape)



#%%

print("\nPAIRWISE SEMANTIC SIMILARITIES\n")

for i, concept_a in enumerate(concepts):
    for j, concept_b in enumerate(concepts):

        if i < j:
            similarity = similarity_matrix[i][j].item()

            print(
                f"{concept_a:12s} <-> {concept_b:12s} "
                f"Similarity: {similarity:.4f}"
            )

#%% [markdown]

#%% ==========================================================
# 6. GEOMETRIC VISUALIZATION
# ============================================================

"""
Human beings cannot directly visualize:

    384-dimensional geometry

So we project the embeddings
into lower dimensions.

------------------------------------------------------------
DIMENSIONALITY REDUCTION
------------------------------------------------------------

We use PCA:

    Principal Component Analysis

PCA attempts to preserve major variance structure.

------------------------------------------------------------
IMPORTANT WARNING
------------------------------------------------------------

The 2D visualization is NOT the true geometry.

It is only a projection.

High-dimensional semantic relationships
may partially disappear during projection.

------------------------------------------------------------
QUESTION
------------------------------------------------------------

Can semantic structure survive dimensional compression?
"""



#%%

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Convert tensor to NumPy
embedding_numpy = embeddings.cpu().numpy()

# Reduce dimensionality
pca = PCA(n_components=2)

reduced_embeddings = pca.fit_transform(embedding_numpy)



#%%

plt.figure(figsize=(12, 10))

for i, word in enumerate(concepts):

    x = reduced_embeddings[i][0]
    y = reduced_embeddings[i][1]

    plt.scatter(x, y)

    plt.text(
        x + 0.02,
        y + 0.02,
        word,
        fontsize=11
    )

plt.title("Semantic Geometry of Real Embeddings")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.grid(True)

plt.show()

#%% [markdown]

#%% ==========================================================
# 7. INTERPRETATION
# ============================================================

"""
OBSERVATION ANALYSIS
--------------------

Carefully inspect the visualization.

Possible patterns:

    • animals cluster together
    • royalty concepts cluster together
    • emotions may cluster
    • scientific domains may align

------------------------------------------------------------
IMPORTANT REALIZATION
------------------------------------------------------------

The model was not manually programmed
with semantic rules.

Instead:

semantic structure emerged automatically
through training.

------------------------------------------------------------
DEEP QUESTION
------------------------------------------------------------

What exactly is being represented?

Is the embedding storing:

    • meaning?
    • statistical structure?
    • contextual predictability?
    • latent semantics?
    • relational topology?

Modern AI research still investigates this deeply.

------------------------------------------------------------
ANOTHER IMPORTANT INSIGHT
------------------------------------------------------------

Embedding spaces are NOT random.

They possess:

    • topology
    • geometry
    • clustering structure
    • relational organization

This is why semantic retrieval works.
"""

#%% [markdown]

#%% ==========================================================
# 8. FAILURE CASES
# ============================================================

"""
REAL SYSTEMS ARE IMPERFECT
--------------------------

Embeddings possess limitations.

------------------------------------------------------------
LIMITATION 1 — POLYSEMY
------------------------------------------------------------

Example:

    bank

Could mean:

    • financial institution
    • river bank

Static embeddings struggle
with contextual ambiguity.

------------------------------------------------------------
LIMITATION 2 — DATASET BIAS
------------------------------------------------------------

Embeddings inherit patterns from training data.

This may introduce:

    • social bias
    • political bias
    • cultural imbalance

------------------------------------------------------------
LIMITATION 3 — PROJECTION DISTORTION
------------------------------------------------------------

2D PCA projections distort
high-dimensional geometry.

Nearby points in 2D may not be
truly nearby in 384D space.

------------------------------------------------------------
LIMITATION 4 — SEMANTIC COMPLEXITY
------------------------------------------------------------

Not all human meaning
can be compressed perfectly into vectors.

------------------------------------------------------------
THINK CRITICALLY
------------------------------------------------------------

Can geometry fully represent meaning?

Or are embeddings only approximations
of semantic structure?
"""

#%% [markdown]

#%% ==========================================================
# 9. SCIENTIFIC INSIGHT
# ============================================================

"""
MOST IMPORTANT INSIGHT
----------------------

Modern AI systems fundamentally operate through:

    structured representation geometry

not symbolic memorization.

------------------------------------------------------------
EMBEDDINGS ENABLE
------------------------------------------------------------

    • semantic search
    • recommendation systems
    • retrieval pipelines
    • contextual reasoning
    • multimodal alignment
    • vector databases
    • RAG systems

------------------------------------------------------------
FOUNDATIONAL REALIZATION
------------------------------------------------------------

Language becomes navigable geometry.

Meaning becomes spatial structure.

Similarity becomes measurable distance.

------------------------------------------------------------
VERY IMPORTANT TRANSITION
------------------------------------------------------------

This step introduces the learner to:

    representation-centric AI thinking

This is one of the most important
conceptual transitions in modern AI.

------------------------------------------------------------
LOOK AHEAD
------------------------------------------------------------

Next steps will investigate:

    • vector arithmetic
    • semantic directions
    • attention dynamics
    • contextual representation shift
    • latent spaces
    • multimodal alignment

We are gradually building toward:

    AI as geometric information systems.
"""

#%% [markdown]

#%% ==========================================================
# 10. EXPLORATION EXERCISES
# ============================================================

"""
EXPLORATION 1
-------------

Add abstract concepts:

    • freedom
    • justice
    • intelligence
    • consciousness

Do abstract concepts form clusters?

------------------------------------------------------------
EXPLORATION 2
-------------

Add emotional concepts:

    • joy
    • grief
    • fear
    • excitement

How does emotional geometry organize?

------------------------------------------------------------
EXPLORATION 3
-------------

Compare scientific disciplines:

    • mathematics
    • biology
    • physics
    • philosophy

Which fields appear geometrically related?

------------------------------------------------------------
EXPLORATION 4
-------------

Try multilingual embeddings.

Do concepts from different languages
occupy nearby semantic regions?

------------------------------------------------------------
EXPLORATION 5
-------------

Replace PCA with:

    • t-SNE
    • UMAP

How does visualization structure change?

------------------------------------------------------------
FINAL QUESTION
------------------------------------------------------------

If modern AI systems fundamentally operate
through representation geometry...

Then what exactly is intelligence?

Pattern matching?

Semantic organization?

Dynamic geometric reasoning?

Or something even deeper?
"""