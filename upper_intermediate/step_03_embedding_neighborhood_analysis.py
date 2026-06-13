#%% [markdown]
# ============================================================
# PHASE 06 — APPLIED REPRESENTATION ANALYSIS
# STEP 03 — EMBEDDING NEIGHBORHOOD ANALYSIS
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
# How does semantic similarity become spatial proximity?
#
# Modern embedding systems organize concepts into:
#
#     geometric neighborhoods
#
# Nearby vectors often correspond to:
#
#     semantically related concepts
#
# This step investigates:
#
#     • nearest-neighbor structure
#     • semantic clustering
#     • local geometry
#     • similarity metrics
#     • neighborhood topology
#
# using REAL pretrained embedding systems.
#
# ============================================================



#%% [markdown]
# ============================================================
# 1. CONCEPTUAL INTRODUCTION
# ============================================================
#
# CONCEPTUAL FOUNDATION
# ---------------------
#
# Embedding systems transform concepts into vectors:
#
#     cat      -> vector
#     dog      -> vector
#     tiger    -> vector
#
# But embeddings are not randomly scattered.
#
# Instead:
#
#     semantically related concepts
#     often occupy nearby geometric regions.
#
# ------------------------------------------------------------
# SEMANTIC NEIGHBORHOODS
# ------------------------------------------------------------
#
# Imagine semantic space as a landscape.
#
# Nearby regions may contain:
#
#     • animals
#     • emotions
#     • scientific concepts
#     • technologies
#     • political ideas
#
# Each concept exists inside:
#
#     a semantic neighborhood.
#
# ------------------------------------------------------------
# IMPORTANT IDEA
# ------------------------------------------------------------
#
# Semantic meaning becomes:
#
#     spatial organization.
#
# This is foundational to:
#
#     • semantic search
#     • retrieval systems
#     • recommendation systems
#     • RAG pipelines
#     • vector databases
#
# ------------------------------------------------------------
# DEEP QUESTION
# ------------------------------------------------------------
#
# If similar meanings become nearby vectors,
#
# then:
#
#     what kind of geometry does language create?
#



#%% [markdown]
# ============================================================
# 2. REPRESENTATION HYPOTHESIS
# ============================================================
#
# REPRESENTATION HYPOTHESIS
# -------------------------
#
# Hypothesis:
#
#     semantically related concepts
#     should form local neighborhoods
#     in embedding space.
#
# Expected observations:
#
#     cat      near dog
#     physics  near chemistry
#     king     near queen
#     software near computer
#
# ------------------------------------------------------------
# LOCAL GEOMETRY
# ------------------------------------------------------------
#
# Instead of studying:
#
#     entire semantic space
#
# we now investigate:
#
#     local semantic structure.
#
# ------------------------------------------------------------
# IMPORTANT QUESTION
# ------------------------------------------------------------
#
# How does the model decide:
#
#     which concepts become neighbors?
#
# Through:
#
#     • co-occurrence statistics?
#     • contextual similarity?
#     • latent semantic structure?
#
# ------------------------------------------------------------
# THINK DEEPLY
# ------------------------------------------------------------
#
# Is meaning itself:
#
#     a neighborhood structure?
#



#%% [markdown]
# ============================================================
# 3. LOAD REAL EMBEDDING SYSTEM
# ============================================================
#
# We now load a REAL pretrained embedding system.
#
# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------
#
# We use:
#
#     all-MiniLM-L6-v2
#
# This model maps text into:
#
#     384-dimensional embedding vectors.
#
# ------------------------------------------------------------
# GLOSSARY OF FUNCTIONS
# ------------------------------------------------------------
#
# SentenceTransformer(...)
#     Loads pretrained embedding system.
#
# model.encode(...)
#     Converts text into embeddings.
#
# cos_sim(...)
#     Computes cosine similarity.
#
# torch.topk(...)
#     Finds highest similarity values.
#
# PCA(...)
#     Reduces dimensionality for visualization.
#
# ------------------------------------------------------------
# IMPORTANT INSIGHT
# ------------------------------------------------------------
#
# The model was not manually programmed
# with semantic rules.
#
# Instead:
#
#     semantic geometry emerged through training.
#



#%%

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

import torch
import numpy as np

# Load pretrained embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding system loaded successfully.")



#%% [markdown]
# ============================================================
# 4. EXTRACT EMBEDDINGS
# ============================================================
#
# We now transform concepts into geometry.
#
# ------------------------------------------------------------
# IMPORTANT IDEA
# ------------------------------------------------------------
#
# Each embedding becomes:
#
#     a point in semantic space.
#
# Similar meanings may occupy:
#
#     nearby regions.
#
# ------------------------------------------------------------
# VOCABULARY DESIGN
# ------------------------------------------------------------
#
# We intentionally include:
#
#     • animals
#     • royalty
#     • emotions
#     • science
#     • technology
#
# to observe:
#
#     neighborhood formation.
#
# ------------------------------------------------------------
# QUESTION
# ------------------------------------------------------------
#
# Will semantic categories naturally cluster?
#



#%%

concepts = [
    "cat",
    "dog",
    "lion",
    "tiger",
    "wolf",
    "king",
    "queen",
    "prince",
    "princess",
    "computer",
    "software",
    "programming",
    "physics",
    "chemistry",
    "biology",
    "happy",
    "sad",
    "angry",
    "joy",
    "fear"
]

# Generate embeddings
embeddings = model.encode(
    concepts,
    convert_to_tensor=True
)

print("Embedding tensor shape:")
print(embeddings.shape)



#%%

# Create embedding lookup dictionary

embedding_lookup = {
    concept: embeddings[i]
    for i, concept in enumerate(concepts)
}

print("Embedding lookup table created.")



#%% [markdown]
# ============================================================
# 5. SEMANTIC SIMILARITY ANALYSIS
# ============================================================
#
# CORE IDEA
# ---------
#
# We now compute:
#
#     pairwise semantic similarity.
#
# ------------------------------------------------------------
# COSINE SIMILARITY
# ------------------------------------------------------------
#
# We use cosine similarity because:
#
#     semantic direction often matters more
#     than vector magnitude.
#
# ------------------------------------------------------------
# MATHEMATICAL FORMULATION
# ------------------------------------------------------------
#
# Cosine similarity:
#



#
# where:
#
#     x · y      = dot product
#     ||x||      = vector magnitude
#     θ          = angle between vectors
#
# ------------------------------------------------------------
# INTERPRETATION
# ------------------------------------------------------------
#
# Values:
#
#      1.0  -> highly similar
#      0.0  -> unrelated
#     -1.0  -> opposite direction
#
# ------------------------------------------------------------
# IMPORTANT INSIGHT
# ------------------------------------------------------------
#
# Nearby vectors often imply:
#
#     shared semantic structure.
#
# ------------------------------------------------------------
# DEEP QUESTION
# ------------------------------------------------------------
#
# Why does angular similarity correlate
# with semantic similarity?
#



#%%

# Compute pairwise similarity matrix

similarity_matrix = cos_sim(
    embeddings,
    embeddings
)

print("Similarity matrix shape:")
print(similarity_matrix.shape)



#%%

print("\nPAIRWISE SEMANTIC SIMILARITIES\n")

for i, concept_a in enumerate(concepts):

    for j, concept_b in enumerate(concepts):

        if i < j:

            similarity = similarity_matrix[i][j].item()

            print(
                f"{concept_a:12s} <-> "
                f"{concept_b:12s} : "
                f"{similarity:.4f}"
            )



#%% [markdown]
# ============================================================
# 6. NEAREST-NEIGHBOR ANALYSIS
# ============================================================
#
# We now investigate:
#
#     local semantic neighborhoods.
#
# ------------------------------------------------------------
# CORE QUESTION
# ------------------------------------------------------------
#
# Given a concept:
#
#     which other concepts occupy nearby regions?
#
# ------------------------------------------------------------
# NEAREST-NEIGHBOR SEARCH
# ------------------------------------------------------------
#
# This is foundational to:
#
#     • semantic search
#     • vector retrieval
#     • recommendation systems
#     • retrieval-augmented generation
#
# ------------------------------------------------------------
# GEOMETRIC INTERPRETATION
# ------------------------------------------------------------
#
# Neighbor retrieval means:
#
#     searching semantic space
#
# for nearby geometric states.
#
# ------------------------------------------------------------
# IMPORTANT QUESTION
# ------------------------------------------------------------
#
# Does intelligence partially emerge through:
#
#     neighborhood traversal?
#



#%%

def find_nearest_neighbors(
    query_word,
    embedding_lookup,
    top_k=5
):
    """
    Find nearest semantic neighbors.

    PARAMETERS
    ----------
    query_word:
        Concept to investigate.

    embedding_lookup:
        Dictionary of embeddings.

    top_k:
        Number of neighbors to return.

    RETURNS
    -------
    Sorted similarity neighbors.

    --------------------------------------------------------
    IMPORTANT IDEA
    --------------------------------------------------------

    We are effectively asking:

        Which regions of semantic space
        surround this concept?
    """

    query_embedding = embedding_lookup[query_word]

    similarities = []

    for word, embedding in embedding_lookup.items():

        similarity = cos_sim(
            query_embedding,
            embedding
        ).item()

        similarities.append((word, similarity))

    similarities = sorted(
        similarities,
        key=lambda x: x[1],
        reverse=True
    )

    return similarities[:top_k]



#%%

query_words = [
    "cat",
    "king",
    "physics",
    "happy"
]

for query in query_words:

    print("\n" + "=" * 60)
    print(f"NEAREST NEIGHBORS FOR: {query}")
    print("=" * 60)

    neighbors = find_nearest_neighbors(
        query,
        embedding_lookup,
        top_k=6
    )

    for word, score in neighbors:

        print(f"{word:12s} Similarity: {score:.4f}")



#%% [markdown]
# ============================================================
# 7. GEOMETRIC VISUALIZATION
# ============================================================
#
# Human beings cannot directly visualize:
#
#     384-dimensional semantic geometry.
#
# So we project embeddings into:
#
#     2D visualization space.
#
# ------------------------------------------------------------
# PCA FOUNDATION
# ------------------------------------------------------------
#
# PCA:
#
#     Principal Component Analysis
#
# identifies directions of maximum variance.
#
# ------------------------------------------------------------
# COVARIANCE MATRIX
# ------------------------------------------------------------
#



#
# ------------------------------------------------------------
# EIGENVECTOR FORMULATION
# ------------------------------------------------------------
#



#
# ------------------------------------------------------------
# PROJECTION OPERATION
# ------------------------------------------------------------
#



#
# ------------------------------------------------------------
# IMPORTANT WARNING
# ------------------------------------------------------------
#
# The 2D visualization is NOT the true geometry.
#
# It is only a lower-dimensional projection.
#
# Some semantic relationships may distort
# during dimensional compression.
#
# ------------------------------------------------------------
# QUESTION
# ------------------------------------------------------------
#
# Can semantic structure survive dimensional reduction?
#



#%%

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Convert embeddings to NumPy
embedding_numpy = embeddings.cpu().numpy()

# PCA projection
pca = PCA(n_components=2)

reduced_embeddings = pca.fit_transform(
    embedding_numpy
)



#%%

plt.figure(figsize=(14, 10))

for i, concept in enumerate(concepts):

    x = reduced_embeddings[i][0]
    y = reduced_embeddings[i][1]

    plt.scatter(x, y)

    plt.text(
        x + 0.02,
        y + 0.02,
        concept,
        fontsize=11
    )

plt.title("Semantic Neighborhood Geometry")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.grid(True)

plt.show()



#%% [markdown]
# ============================================================
# 8. INTERPRETATION
# ============================================================
#
# OBSERVATION ANALYSIS
# --------------------
#
# Examine the visualization carefully.
#
# Possible observations:
#
#     • animals cluster together
#     • royalty concepts cluster
#     • sciences align nearby
#     • emotions form neighborhoods
#
# ------------------------------------------------------------
# IMPORTANT REALIZATION
# ------------------------------------------------------------
#
# The model was NOT manually given:
#
#     semantic categories.
#
# Instead:
#
#     structure emerged automatically
#     during representation learning.
#
# ------------------------------------------------------------
# LOCAL GEOMETRY
# ------------------------------------------------------------
#
# Embedding spaces may possess:
#
#     • local neighborhoods
#     • semantic regions
#     • cluster topology
#     • relational continuity
#
# ------------------------------------------------------------
# VERY IMPORTANT INSIGHT
# ------------------------------------------------------------
#
# Modern retrieval systems work because:
#
#     semantic meaning becomes searchable geometry.
#
# ------------------------------------------------------------
# DEEP QUESTION
# ------------------------------------------------------------
#
# Is semantic memory fundamentally:
#
#     geometric organization?
#



#%% [markdown]
# ============================================================
# 9. FAILURE CASES
# ============================================================
#
# REAL SYSTEMS ARE IMPERFECT
# --------------------------
#
# Embedding neighborhoods possess limitations.
#
# ------------------------------------------------------------
# LIMITATION 1 — CONTEXT AMBIGUITY
# ------------------------------------------------------------
#
# Example:
#
#     bank
#
# could mean:
#
#     • financial institution
#     • river bank
#
# Static embeddings struggle with ambiguity.
#
# ------------------------------------------------------------
# LIMITATION 2 — DATASET BIAS
# ------------------------------------------------------------
#
# Embeddings inherit patterns from training data.
#
# This may introduce:
#
#     • stereotypes
#     • cultural imbalance
#     • political bias
#
# ------------------------------------------------------------
# LIMITATION 3 — HIGH-DIMENSIONAL DISTORTION
# ------------------------------------------------------------
#
# Neighborhood relationships in 2D
# may differ from true 384D geometry.
#
# ------------------------------------------------------------
# LIMITATION 4 — SEMANTIC COMPLEXITY
# ------------------------------------------------------------
#
# Human meaning may not perfectly map
# into geometric proximity.
#
# ------------------------------------------------------------
# THINK CRITICALLY
# ------------------------------------------------------------
#
# Are semantic neighborhoods sufficient
# to represent human understanding?
#



#%% [markdown]
# ============================================================
# 10. SCIENTIFIC INSIGHT
# ============================================================
#
# MOST IMPORTANT INSIGHT
# ----------------------
#
# Modern AI systems organize meaning through:
#
#     geometric neighborhood structure.
#
# ------------------------------------------------------------
# EMBEDDING SPACES ENABLE
# ------------------------------------------------------------
#
#     • semantic retrieval
#     • recommendation systems
#     • vector databases
#     • semantic search
#     • contextual routing
#     • retrieval-augmented generation
#
# ------------------------------------------------------------
# FOUNDATIONAL REALIZATION
# ------------------------------------------------------------
#
# Meaning becomes:
#
#     navigable geometric structure.
#
# ------------------------------------------------------------
# IMPORTANT TRANSITION
# ------------------------------------------------------------
#
# This step introduces a major idea:
#
#     intelligence may partially operate
#     through neighborhood traversal
#     inside representation spaces.
#
# ------------------------------------------------------------
# LOOK AHEAD
# ------------------------------------------------------------
#
# Next steps investigate:
#
#     • attention maps
#     • contextual representation shifts
#     • transformer interaction dynamics
#     • latent manifolds
#     • multimodal alignment
#
# We are progressively uncovering:
#
#     AI as dynamic representation geometry.
#



#%% [markdown]
# ============================================================
# 11. EXPLORATION EXERCISES
# ============================================================
#
# EXPLORATION 1
# -------------
#
# Add abstract concepts:
#
#     • freedom
#     • justice
#     • intelligence
#     • consciousness
#
# Do abstract ideas form neighborhoods?
#
# ------------------------------------------------------------
# EXPLORATION 2
# -------------
#
# Add political concepts:
#
#     • democracy
#     • socialism
#     • capitalism
#
# How does political geometry organize?
#
# ------------------------------------------------------------
# EXPLORATION 3
# -------------
#
# Replace cosine similarity with:
#
#     • Euclidean distance
#     • Manhattan distance
#
# Does neighborhood structure change?
#
# ------------------------------------------------------------
# EXPLORATION 4
# -------------
#
# Replace PCA with:
#
#     • t-SNE
#     • UMAP
#
# Which visualization preserves clusters better?
#
# ------------------------------------------------------------
# EXPLORATION 5
# -------------
#
# Use sentence embeddings instead of words.
#
# Example:
#
#     "The cat slept."
#     "The dog rested."
#
# Do sentence neighborhoods emerge?
#
# ------------------------------------------------------------
# FINAL QUESTION
# ------------------------------------------------------------
#
# If semantic meaning becomes neighborhood structure...
#
# then could cognition itself emerge through:
#
#     navigation across representation manifolds?
#
# ============================================================