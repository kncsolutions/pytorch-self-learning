# ============================================================
# PHASE 06 — APPLIED REPRESENTATION ANALYSIS
# STEP 07 — CONTEXTUAL REPRESENTATION SHIFT
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
# How does context change representation meaning?
#
# Earlier steps studied:
#
#     • embeddings
#     • attention maps
#     • token interaction systems
#     • head specialization
#
# Now we investigate something deeper:
#
#     contextual semantic transformation.
#
# This step studies:
#
#     • contextual embeddings
#     • semantic reinterpretation
#     • context-dependent meaning
#     • representation shift
#     • contextual geometry
#
# using REAL transformer systems.
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
# Static embedding systems assign:
#
#     one vector per word.
#
# Example:
#
#     bank → fixed vector
#
# But human language is contextual.
#
# ------------------------------------------------------------
# EXAMPLE
# ------------------------------------------------------------
#
# Sentence 1:
#
#     "The bank approved the loan."
#
# Sentence 2:
#
#     "The boat reached the bank."
#
# ------------------------------------------------------------
# IMPORTANT QUESTION
# ------------------------------------------------------------
#
# Does:
#
#     "bank"
#
# possess the same meaning
# in both sentences?
#
# Obviously:
#
#     no.
#
# ------------------------------------------------------------
# TRANSFORMER INSIGHT
# ------------------------------------------------------------
#
# Transformers dynamically modify:
#
#     token representations
#
# according to:
#
#     surrounding context.
#
# ------------------------------------------------------------
# CORE IDEA
# ------------------------------------------------------------
#
# Meaning is NOT static.
#
# Meaning emerges through:
#
#     contextual interaction.
#
# ------------------------------------------------------------
# DEEP QUESTION
# ------------------------------------------------------------
#
# Could semantic meaning itself be:
#
#     a contextual geometric state?
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
#     identical words appearing in
#     different contexts
#
# should produce:
#
#     different contextual embeddings.
#
# Expected observations:
#
#     • contextual meaning shifts
#     • embeddings separate geometrically
#     • semantic neighborhoods change
#     • contextual interaction modifies representation
#
# ------------------------------------------------------------
# IMPORTANT IDEA
# ------------------------------------------------------------
#
# Context may continuously reshape:
#
#     semantic geometry.
#
# ------------------------------------------------------------
# THINK DEEPLY
# ------------------------------------------------------------
#
# Is meaning fundamentally:
#
#     relational rather than isolated?
#



#%% [markdown]
# ============================================================
# 3. LOAD REAL TRANSFORMER SYSTEM
# ============================================================
#
# We now load a REAL contextual transformer.
#
# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------
#
# We use:
#
#     BERT
#
# because it produces:
#
#     contextualized token embeddings.
#
# ------------------------------------------------------------
# GLOSSARY OF FUNCTIONS
# ------------------------------------------------------------
#
# BertTokenizer.from_pretrained(...)
#     Loads tokenizer.
#
# BertModel.from_pretrained(...)
#     Loads transformer model.
#
# tokenizer(...)
#     Converts text into token IDs.
#
# output_hidden_states=True
#     Returns contextual hidden states.
#
# cosine_similarity(...)
#     Measures geometric similarity.
#
# PCA(...)
#     Reduces dimensionality for visualization.
#
# ------------------------------------------------------------
# IMPORTANT INSIGHT
# ------------------------------------------------------------
#
# Transformer embeddings are NOT fixed.
#
# They evolve dynamically:
#
#     sentence-by-sentence.
#



#%%

import torch

from transformers import (
    BertTokenizer,
    BertModel
)

from sklearn.metrics.pairwise import cosine_similarity

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased"
)

# Load transformer model
model = BertModel.from_pretrained(
    "bert-base-uncased",
    output_hidden_states=True
)

print("Transformer system loaded successfully.")



#%% [markdown]
# ============================================================
# 4. CONTEXTUAL SENTENCES
# ============================================================
#
# We intentionally choose:
#
#     semantically ambiguous words.
#
# ------------------------------------------------------------
# EXAMPLE WORD
# ------------------------------------------------------------
#
#     bank
#
# Context 1:
#
#     financial institution
#
# Context 2:
#
#     river edge
#
# ------------------------------------------------------------
# IMPORTANT IDEA
# ------------------------------------------------------------
#
# The SAME token should develop:
#
#     different contextual geometry.
#
# ------------------------------------------------------------
# QUESTION
# ------------------------------------------------------------
#
# How much can context reshape meaning?
#



#%%

sentence_1 = "The bank approved the loan."
sentence_2 = "The boat reached the bank."

print("Sentence 1:")
print(sentence_1)

print("\nSentence 2:")
print(sentence_2)



#%% [markdown]
# ============================================================
# 5. TOKENIZATION AND CONTEXTUAL PROCESSING
# ============================================================
#
# Transformers first convert text into:
#
#     contextual token sequences.
#
# ------------------------------------------------------------
# NOTE ON SPECIAL TOKENS
# ------------------------------------------------------------
#
# [CLS]
#
#     global semantic aggregation token.
#
# [SEP]
#
#     sequence boundary token.
#
# ------------------------------------------------------------
# IMPORTANT IDEA
# ------------------------------------------------------------
#
# Every token representation depends on:
#
#     surrounding contextual interactions.
#



#%%

# Tokenize both sentences

inputs_1 = tokenizer(
    sentence_1,
    return_tensors="pt"
)

inputs_2 = tokenizer(
    sentence_2,
    return_tensors="pt"
)

# Convert IDs back to readable tokens

tokens_1 = tokenizer.convert_ids_to_tokens(
    inputs_1["input_ids"][0]
)

tokens_2 = tokenizer.convert_ids_to_tokens(
    inputs_2["input_ids"][0]
)

print("\nTOKENS — SENTENCE 1\n")
print(tokens_1)

print("\nTOKENS — SENTENCE 2\n")
print(tokens_2)



#%% [markdown]
# ============================================================
# 6. CONTEXTUAL REPRESENTATION MATHEMATICS
# ============================================================
#
# CORE IDEA
# ---------
#
# Transformer representations evolve through:
#
#     contextual interaction.
#
# ============================================================
# INITIAL EMBEDDINGS
# ============================================================
#
# Initially:
#
#     tokens receive embedding vectors.
#
# Example:
#
#     x_bank
#
# ------------------------------------------------------------
# IMPORTANT LIMITATION
# ------------------------------------------------------------
#
# Initial embeddings are:
#
#     context-independent.
#
# ============================================================
# CONTEXTUAL TRANSFORMATION
# ============================================================
#
# Through attention interaction:
#
# representations continuously update.
#
# ------------------------------------------------------------
# ATTENTION OPERATION
# ------------------------------------------------------------
#



#
# ------------------------------------------------------------
# CONTEXTUAL REPRESENTATION UPDATE
# ------------------------------------------------------------
#



#
# ------------------------------------------------------------
# INTERPRETATION
# ------------------------------------------------------------
#
# Token representations become:
#
#     weighted contextual combinations
#
# of surrounding tokens.
#
# ============================================================
# CONTEXTUAL EMBEDDING FUNCTION
# ============================================================
#
# Transformer representation:
#



#
# where:
#
#     h_i^(l)
#
# represents:
#
#     token i representation
#     at layer l.
#
# ------------------------------------------------------------
# IMPORTANT INSIGHT
# ------------------------------------------------------------
#
# A token representation depends on:
#
#     all surrounding representations.
#
# ============================================================
# CONTEXTUAL SIMILARITY
# ============================================================
#
# We compare contextual embeddings using:
#
#     cosine similarity.
#
# ------------------------------------------------------------
# FORMULATION
# ------------------------------------------------------------
#


#
# ------------------------------------------------------------
# INTERPRETATION
# ------------------------------------------------------------
#
# High similarity:
#
#     contextual meanings remain similar.
#
# Low similarity:
#
#     contextual meanings diverge.
#
# ------------------------------------------------------------
# DEEP QUESTION
# ------------------------------------------------------------
#
# Is meaning fundamentally:
#
#     contextual geometry?
#



#%% [markdown]
# ============================================================
# 7. EXTRACT CONTEXTUAL REPRESENTATIONS
# ============================================================
#
# We now pass both sentences
# through the transformer.
#
# ------------------------------------------------------------
# IMPORTANT IDEA
# ------------------------------------------------------------
#
# Even though:
#
#     "bank"
#
# appears in both sentences,
#
# contextual interaction should produce:
#
#     different representations.
#



#%%

# Forward pass through transformer

with torch.no_grad():

    outputs_1 = model(**inputs_1)
    outputs_2 = model(**inputs_2)

# Extract final hidden states

hidden_1 = outputs_1.last_hidden_state
hidden_2 = outputs_2.last_hidden_state

print("Hidden state shape — Sentence 1:")
print(hidden_1.shape)

print("\nHidden state shape — Sentence 2:")
print(hidden_2.shape)



#%% [markdown]
# ============================================================
# 8. IDENTIFY TARGET TOKEN
# ============================================================
#
# We now locate:
#
#     "bank"
#
# inside both token sequences.
#
# ------------------------------------------------------------
# IMPORTANT IDEA
# ------------------------------------------------------------
#
# We will compare:
#
#     contextual embeddings
#
# for the SAME token
#
# across different contexts.
#



#%%

# Locate token index

bank_index_1 = tokens_1.index("bank")
bank_index_2 = tokens_2.index("bank")

print("Bank token index — Sentence 1:")
print(bank_index_1)

print("\nBank token index — Sentence 2:")
print(bank_index_2)



#%% [markdown]
# ============================================================
# 9. EXTRACT CONTEXTUAL EMBEDDINGS
# ============================================================
#
# We now extract:
#
#     contextual representations
#
# for:
#
#     "bank"
#
# from both sentences.
#
# ------------------------------------------------------------
# IMPORTANT IDEA
# ------------------------------------------------------------
#
# These embeddings now contain:
#
#     contextual semantic information.
#



#%%

# Extract contextual embeddings

bank_embedding_1 = (
    hidden_1[0, bank_index_1]
    .cpu()
    .numpy()
)

bank_embedding_2 = (
    hidden_2[0, bank_index_2]
    .cpu()
    .numpy()
)

print("Contextual embedding shape:")
print(bank_embedding_1.shape)



#%% [markdown]
# ============================================================
# 10. CONTEXTUAL SIMILARITY ANALYSIS
# ============================================================
#
# We now measure:
#
#     geometric similarity
#
# between:
#
#     contextual representations.
#
# ------------------------------------------------------------
# IMPORTANT IDEA
# ------------------------------------------------------------
#
# If context strongly changes meaning,
#
# then:
#
#     similarity should decrease.
#



#%%

# Compute cosine similarity

similarity = cosine_similarity(
    [bank_embedding_1],
    [bank_embedding_2]
)[0][0]

print("\nCONTEXTUAL REPRESENTATION SIMILARITY\n")

print(
    f"Similarity between contextual "
    f"'bank' embeddings: {similarity:.4f}"
)



#%% [markdown]
# ============================================================
# 11. CONTEXTUAL GEOMETRY VISUALIZATION
# ============================================================
#
# Human beings cannot directly visualize:
#
#     768-dimensional contextual embeddings.
#
# So we project representations into:
#
#     2D semantic space.
#
# ------------------------------------------------------------
# PCA FOUNDATION
# ------------------------------------------------------------
#
# PCA identifies:
#
#     dominant variance directions.
#
# ------------------------------------------------------------
# IMPORTANT QUESTION
# ------------------------------------------------------------
#
# Will contextual meanings separate geometrically?
#



#%%

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

# Stack contextual embeddings

embedding_matrix = np.vstack([
    bank_embedding_1,
    bank_embedding_2
])

# Apply PCA projection

pca = PCA(n_components=2)

reduced_embeddings = pca.fit_transform(
    embedding_matrix
)



#%%

labels = [
    "bank (financial)",
    "bank (river)"
]

plt.figure(figsize=(10, 8))

for i, label in enumerate(labels):

    x = reduced_embeddings[i][0]
    y = reduced_embeddings[i][1]

    plt.scatter(x, y)

    plt.text(
        x + 0.01,
        y + 0.01,
        label,
        fontsize=12
    )

plt.title(
    "Contextual Representation Shift"
)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.grid(True)

plt.show()



#%% [markdown]
# ============================================================
# 12. CONTEXTUAL INTERPRETATION
# ============================================================
#
# OBSERVATION ANALYSIS
# --------------------
#
# Examine the similarity score carefully.
#
# Possible observations:
#
#     • contextual embeddings diverge
#     • semantic geometry shifts
#     • meaning changes with context
#
# ------------------------------------------------------------
# IMPORTANT REALIZATION
# ------------------------------------------------------------
#
# Transformers do NOT store:
#
#     fixed word meanings.
#
# Instead:
#
#     meaning dynamically emerges
#     through contextual interaction.
#
# ------------------------------------------------------------
# VERY IMPORTANT INSIGHT
# ------------------------------------------------------------
#
# Semantic representations behave like:
#
#     adaptive contextual states.
#
# ------------------------------------------------------------
# DEEP QUESTION
# ------------------------------------------------------------
#
# Is human meaning itself:
#
#     dynamically contextual?
#



#%% [markdown]
# ============================================================
# 13. CONTEXTUAL REPRESENTATION EVOLUTION
# ============================================================
#
# Earlier AI systems relied heavily on:
#
#     static embeddings.
#
# Transformers introduced:
#
#     contextual representation dynamics.
#
# ------------------------------------------------------------
# IMPORTANT TRANSITION
# ------------------------------------------------------------
#
# Before transformers:
#
#     word → fixed vector
#
# After transformers:
#
#     word + context → contextual vector
#
# ------------------------------------------------------------
# FOUNDATIONAL REALIZATION
# ------------------------------------------------------------
#
# Meaning is NOT stored statically.
#
# Meaning evolves through:
#
#     interaction with surrounding representations.
#
# ------------------------------------------------------------
# IMPORTANT INSIGHT
# ------------------------------------------------------------
#
# Context acts like:
#
#     a semantic transformation field.
#



#%% [markdown]
# ============================================================
# 14. FAILURE CASES
# ============================================================
#
# REAL SYSTEMS ARE IMPERFECT
# --------------------------
#
# Contextual embeddings possess limitations.
#
# ------------------------------------------------------------
# LIMITATION 1 — CONTEXT WINDOW
# ------------------------------------------------------------
#
# Transformers only observe:
#
#     finite contextual regions.
#
# ------------------------------------------------------------
# LIMITATION 2 — SEMANTIC AMBIGUITY
# ------------------------------------------------------------
#
# Some meanings remain:
#
#     difficult to separate cleanly.
#
# ------------------------------------------------------------
# LIMITATION 3 — DATASET BIAS
# ------------------------------------------------------------
#
# Contextual representations inherit:
#
#     training-data patterns.
#
# ------------------------------------------------------------
# LIMITATION 4 — VISUALIZATION DISTORTION
# ------------------------------------------------------------
#
# PCA compresses:
#
#     high-dimensional geometry.
#
# ------------------------------------------------------------
# THINK CRITICALLY
# ------------------------------------------------------------
#
# Can contextual embeddings fully capture:
#
#     human semantic understanding?
#



#%% [markdown]
# ============================================================
# 15. SCIENTIFIC INSIGHT
# ============================================================
#
# MOST IMPORTANT INSIGHT
# ----------------------
#
# Transformers fundamentally operate through:
#
#     contextual representation evolution.
#
# ------------------------------------------------------------
# CONTEXTUAL EMBEDDINGS ENABLE
# ------------------------------------------------------------
#
#     • semantic disambiguation
#     • contextual reasoning
#     • adaptive meaning formation
#     • dynamic representation routing
#     • contextual retrieval systems
#
# ------------------------------------------------------------
# FOUNDATIONAL REALIZATION
# ------------------------------------------------------------
#
# Meaning behaves like:
#
#     dynamic geometry
#
# rather than:
#
#     static symbolic storage.
#
# ------------------------------------------------------------
# IMPORTANT TRANSITION
# ------------------------------------------------------------
#
# We are now studying:
#
#     adaptive semantic systems
#
# rather than:
#
#     fixed representation systems.
#
# ------------------------------------------------------------
# LOOK AHEAD
# ------------------------------------------------------------
#
# Next steps investigate:
#
#     • layerwise representation evolution
#     • semantic abstraction growth
#     • latent manifolds
#     • multimodal semantic alignment
#
# We are progressively uncovering:
#
#     AI as contextual geometric cognition.
#



#%% [markdown]
# ============================================================
# 16. EXPLORATION EXERCISES
# ============================================================
#
# EXPLORATION 1
# -------------
#
# Try additional ambiguous words:
#
#     • bat
#     • spring
#     • light
#     • mouse
#
# Does contextual geometry shift similarly?
#
# ------------------------------------------------------------
# EXPLORATION 2
# -------------
#
# Compare:
#
#     short vs long contexts.
#
# Does stronger context produce:
#
#     larger representation shifts?
#
# ------------------------------------------------------------
# EXPLORATION 3
# -------------
#
# Compare:
#
#     early vs late transformer layers.
#
# Where does semantic disambiguation emerge?
#
# ------------------------------------------------------------
# EXPLORATION 4
# -------------
#
# Replace BERT with:
#
#     • RoBERTa
#     • GPT-2
#     • DistilBERT
#
# Does contextual geometry change?
#
# ------------------------------------------------------------
# EXPLORATION 5
# -------------
#
# Visualize:
#
#     contextual neighborhoods
#
# around ambiguous words.
#
# ------------------------------------------------------------
# FINAL QUESTION
# ------------------------------------------------------------
#
# If meaning emerges through:
#
#     contextual representation evolution,
#
# then could cognition fundamentally be:
#
#     dynamic geometric reinterpretation?
#
# ============================================================