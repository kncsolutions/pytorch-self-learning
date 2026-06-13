#%% [markdown]
# ============================================================
# PHASE 06 — APPLIED REPRESENTATION ANALYSIS
# STEP 02 — VECTOR ARITHMETIC WITH REAL EMBEDDINGS
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
# Can semantic relationships emerge through vector operations?
#
# Example:
#
#     king - man + woman ≈ queen
#
# This is one of the most famous observations
# in representation learning.
#
# But WHY does this happen?
#
# Are embeddings merely storing meanings?
#
# Or are they learning:
#
#     relational geometry?
#
#%% [markdown]
# ------------------------------------------------------------
# CORE IDEA
# ------------------------------------------------------------
#
# Embeddings may encode:
#
#     • semantic similarity
#     • relational directions
#     • conceptual transformations
#     • latent semantic operations
#
# In this step we investigate:
#
#     • vector arithmetic
#     • semantic directions
#     • relational geometry
#     • analogy structure
#     • geometric transformations
#
# using REAL pretrained embedding systems.
#
# ============================================================

#%% [markdown]

#==========================================================
# 1. CONCEPTUAL INTRODUCTION
# ============================================================

"""
CONCEPTUAL FOUNDATION
---------------------

Suppose we represent concepts as vectors:

    king   -> vector
    man    -> vector
    woman  -> vector

Now consider:

    king - man + woman

Why should this operation produce something meaningful?

This seems surprising.

Yet embedding systems often produce:

    queen

as the nearest semantic result.

------------------------------------------------------------
IMPORTANT TRANSITION
------------------------------------------------------------

Embeddings are not merely storing objects.

They may encode:

    relationships as directions.

------------------------------------------------------------
RELATIONAL GEOMETRY
------------------------------------------------------------

Example:

    king -> queen

may represent a semantic direction related to:

    masculinity -> femininity

Similarly:

    man -> woman

may occupy a similar geometric direction.

This suggests:

    semantic relationships become vector fields.

------------------------------------------------------------
DEEP QUESTION
------------------------------------------------------------

If semantic relationships become directions,

then can reasoning itself emerge
through geometric transformation?
"""

#%% [markdown]

#==========================================================
# 2. REPRESENTATION HYPOTHESIS
# ============================================================

"""
REPRESENTATION HYPOTHESIS
-------------------------

Hypothesis:

    semantic relationships may become
    approximately linear inside embedding space.

Expected observations:

    king - man + woman ≈ queen

    paris - france + italy ≈ rome

    walking - walk + swim ≈ swimming

------------------------------------------------------------
WHAT ARE WE TESTING?
------------------------------------------------------------

We are NOT testing memorization.

We are testing whether:

    relational structure becomes geometry.

------------------------------------------------------------
IMPORTANT IDEA
------------------------------------------------------------

If multiple semantic relationships align geometrically,

then embedding spaces may possess:

    structured semantic directions.

------------------------------------------------------------
THINK DEEPLY
------------------------------------------------------------

Why should language relationships become linear?

Is this:

    • statistical compression?
    • latent structure emergence?
    • optimization artifact?
    • semantic organization?
"""


#%% [markdown]
#==========================================================
# 3. LOAD REAL EMBEDDING SYSTEM
# ============================================================

"""
We now load a REAL embedding system.

------------------------------------------------------------
MODEL SELECTION
------------------------------------------------------------

We use:

    all-MiniLM-L6-v2

This transformer-based embedding model maps text into:

    384-dimensional semantic vectors.

------------------------------------------------------------
GLOSSARY OF FUNCTIONS
------------------------------------------------------------

SentenceTransformer(...)
    Loads pretrained embedding model.

model.encode(...)
    Converts text into embedding vectors.

torch.norm(...)
    Computes vector magnitude.

torch.topk(...)
    Returns top-k highest values.

------------------------------------------------------------
IMPORTANT INSIGHT
------------------------------------------------------------

The model was never explicitly programmed with:

    "king - man + woman = queen"

Yet relational geometry may still emerge.
"""



#%%

from sentence_transformers import SentenceTransformer
import torch

# Load pretrained embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Embedding system loaded successfully.")

#%% [markdown]

#==========================================================
# 4. EXTRACT EMBEDDINGS
# ============================================================

"""
We now convert language into geometric vectors.

------------------------------------------------------------
IMPORTANT IDEA
------------------------------------------------------------

Each word becomes:

    a coordinate in semantic space.

But now we investigate something deeper:

    vector operations between meanings.

------------------------------------------------------------
QUESTION
------------------------------------------------------------

Can semantic transformations
become measurable geometric directions?
"""



#%%

vocabulary = [
    "king",
    "queen",
    "man",
    "woman",
    "prince",
    "princess",
    "boy",
    "girl",
    "paris",
    "france",
    "rome",
    "italy",
    "walk",
    "walking",
    "swim",
    "swimming",
    "computer",
    "software",
]

# Extract embeddings
embeddings = model.encode(
    vocabulary,
    convert_to_tensor=True
)

print("Embedding tensor shape:")
print(embeddings.shape)



#%%

# Create lookup dictionary

embedding_lookup = {
    word: embeddings[i]
    for i, word in enumerate(vocabulary)
}

print("Embedding lookup table created.")

#%% [markdown]

#==========================================================
# 5. VECTOR ARITHMETIC ANALYSIS
# ============================================================

"""
CORE IDEA
---------

We now perform semantic vector arithmetic.

------------------------------------------------------------
CLASSIC ANALOGY
------------------------------------------------------------

    king - man + woman

The hypothesis:

    this vector should move toward "queen"

------------------------------------------------------------
GEOMETRIC INTERPRETATION
------------------------------------------------------------

Suppose:

    king -> queen

represents a semantic transformation.

Then:

    removing "man"
    adding "woman"

should shift the representation
toward feminine royalty.

------------------------------------------------------------
IMPORTANT QUESTION
------------------------------------------------------------

Are embeddings learning:

    concepts?

Or:

    transformations between concepts?
"""



#%%

from sentence_transformers.util import cos_sim



def normalize(vector):
    """
    Normalize vector magnitude.

    WHY NORMALIZE?
    --------------

    We want comparison based on direction,
    not vector length.

    This is important because:

        semantic similarity often depends more
        on angular structure than magnitude.
    """

    return vector / torch.norm(vector)



#%%

# ------------------------------------------------------------
# CLASSIC SEMANTIC ANALOGY
# ------------------------------------------------------------

king = embedding_lookup["king"]
man = embedding_lookup["man"]
woman = embedding_lookup["woman"]

# Semantic arithmetic
result_vector = king - man + woman

# Normalize
result_vector = normalize(result_vector)

print("Vector arithmetic completed.")



#%%

"""
We now compare the generated vector
against all vocabulary embeddings.

The nearest embedding may reveal:

    emergent relational structure.
"""



#%%

similarities = []

for word in vocabulary:

    candidate_vector = normalize(embedding_lookup[word])

    similarity = cos_sim(
        result_vector,
        candidate_vector
    ).item()

    similarities.append((word, similarity))

# Sort by similarity
similarities = sorted(
    similarities,
    key=lambda x: x[1],
    reverse=True
)



#%%

print("\nTOP SEMANTIC MATCHES\n")

for word, score in similarities[:10]:

    print(f"{word:12s} Similarity: {score:.4f}")

#%% [markdown]

#==========================================================
# 6. GEOMETRIC VISUALIZATION
# ============================================================

"""
We now visualize relational geometry.

------------------------------------------------------------
GOAL
------------------------------------------------------------

We want to observe:

    • semantic clusters
    • relational directions
    • geometric organization

------------------------------------------------------------
DIMENSIONALITY REDUCTION
------------------------------------------------------------

We use PCA to project high-dimensional embeddings
into 2D visualization space.

------------------------------------------------------------
PCA FOUNDATION
------------------------------------------------------------

PCA computes directions of maximum variance.

Covariance matrix:

"""


"""
Eigenvalue problem:
"""



"""
Projection operation:
"""


"""
IMPORTANT:

The 2D projection is NOT the true geometry.

It is only a compressed approximation
of higher-dimensional semantic structure.
"""



#%%

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Convert tensor to NumPy
embedding_numpy = embeddings.cpu().numpy()

# PCA projection
pca = PCA(n_components=2)

reduced_embeddings = pca.fit_transform(embedding_numpy)



#%%

plt.figure(figsize=(12, 10))

for i, word in enumerate(vocabulary):

    x = reduced_embeddings[i][0]
    y = reduced_embeddings[i][1]

    plt.scatter(x, y)

    plt.text(
        x + 0.02,
        y + 0.02,
        word,
        fontsize=11
    )

plt.title("Semantic Geometry and Relational Structure")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.grid(True)

plt.show()


#%% [markdown]
#==========================================================
# 7. INTERPRETATION
# ============================================================

"""
OBSERVATION ANALYSIS
--------------------

Did the arithmetic produce:

    queen

or something nearby?

If so, this suggests:

    semantic relationships may become geometric operations.

------------------------------------------------------------
IMPORTANT REALIZATION
------------------------------------------------------------

The embedding model was NOT explicitly taught:

    algebraic semantic rules.

Instead:

relational structure emerged during training.

------------------------------------------------------------
RELATIONAL VECTOR FIELDS
------------------------------------------------------------

Embedding spaces may encode:

    • gender direction
    • tense direction
    • geography direction
    • hierarchy direction

This means:

    semantic transformations may become geometric motion.

------------------------------------------------------------
DEEP QUESTION
------------------------------------------------------------

Could reasoning itself emerge
through transformations in latent space?

Modern AI research increasingly investigates this.
"""

#%% [markdown]

#==========================================================
# 8. FAILURE CASES
# ============================================================

"""
REAL SYSTEMS ARE IMPERFECT
--------------------------

Vector arithmetic does NOT always work perfectly.

------------------------------------------------------------
LIMITATION 1 — NONLINEAR SEMANTICS
------------------------------------------------------------

Human meaning is highly nonlinear.

Not all relationships can be represented
through linear transformations.

------------------------------------------------------------
LIMITATION 2 — CONTEXT DEPENDENCE
------------------------------------------------------------

Words change meaning depending on context.

Example:

    bank

could mean:

    • finance
    • river edge

Static arithmetic struggles with ambiguity.

------------------------------------------------------------
LIMITATION 3 — DATASET BIAS
------------------------------------------------------------

Embedding geometry reflects training data.

This may introduce:

    • social bias
    • cultural imbalance
    • stereotype amplification

------------------------------------------------------------
LIMITATION 4 — LIMITED VOCABULARY
------------------------------------------------------------

Our toy vocabulary is tiny.

Real embedding systems operate
over millions of concepts.

------------------------------------------------------------
THINK CRITICALLY
------------------------------------------------------------

Are semantic relationships truly linear?

Or are we observing only approximate local structure?
"""

#%% [markdown]

#==========================================================
# 9. SCIENTIFIC INSIGHT
# ============================================================

"""
MOST IMPORTANT INSIGHT
----------------------

Embeddings may encode:

    relational geometry

not merely isolated meanings.

------------------------------------------------------------
THIS CHANGES EVERYTHING
------------------------------------------------------------

Modern AI systems may operate through:

    geometric transformation of representations

rather than symbolic rule execution.

------------------------------------------------------------
VECTOR ARITHMETIC SUGGESTS
------------------------------------------------------------

    • relationships become directions
    • transformations become motion
    • analogies become geometry
    • reasoning may become latent navigation

------------------------------------------------------------
FOUNDATIONAL TRANSITION
------------------------------------------------------------

This step introduces a critical idea:

    representation spaces are structured systems.

Not random vector containers.

------------------------------------------------------------
LOOK AHEAD
------------------------------------------------------------

Next steps investigate:

    • semantic neighborhoods
    • contextual embedding shifts
    • attention routing
    • latent manifolds
    • multimodal geometry

We are gradually uncovering:

    AI as geometric information processing.
"""

#%% [markdown]

#==========================================================
# 10. EXPLORATION EXERCISES
# ============================================================

"""
EXPLORATION 1
-------------

Try additional analogies:

    paris - france + italy
    walking - walk + swim
    doctor - hospital + school

What structures emerge?

------------------------------------------------------------
EXPLORATION 2
-------------

Investigate opposite relationships:

    happy vs sad
    hot vs cold

Do opposites occupy opposite directions?

------------------------------------------------------------
EXPLORATION 3
-------------

Use sentence embeddings instead of words.

Example:

    "The cat sat on the mat."
    "A dog rested on the floor."

Do relational operations still work?

------------------------------------------------------------
EXPLORATION 4
-------------

Replace PCA with:

    • t-SNE
    • UMAP

Does relational geometry become clearer?

------------------------------------------------------------
EXPLORATION 5
-------------

Try multilingual embeddings.

Does:

    king - english + french

move toward French semantic regions?

------------------------------------------------------------
FINAL QUESTION
------------------------------------------------------------

If relationships themselves become geometry...

Then could intelligence emerge through:

    navigation inside representation space?
"""