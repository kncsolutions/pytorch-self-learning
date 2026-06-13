# ============================================================
# PHASE 06 — APPLIED REPRESENTATION ANALYSIS
# STEP 04 — ATTENTION MAP VISUALIZATION
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
# How do transformers dynamically redistribute attention?
#
# Unlike static embeddings:
#
#     transformer representations are contextual.
#
# Tokens continuously influence each other through:
#
#     attention interactions.
#
# This step investigates:
#
#     • attention maps
#     • token interactions
#     • contextual information flow
#     • attention distributions
#     • dynamic representation routing
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
# Earlier steps investigated:
#
#     static semantic geometry.
#
# But transformers introduce something deeper:
#
#     dynamic contextual interaction.
#
# ------------------------------------------------------------
# THE LIMITATION OF STATIC EMBEDDINGS
# ------------------------------------------------------------
#
# Static embeddings assign:
#
#     one vector per word.
#
# Example:
#
#     bank
#
# always receives:
#
#     same representation.
#
# But human language is contextual:
#
#     river bank
#     financial bank
#
# possess different meanings.
#
# ------------------------------------------------------------
# TRANSFORMER SOLUTION
# ------------------------------------------------------------
#
# Transformers dynamically update representations
# through:
#
#     attention interactions.
#
# Tokens "look at" other tokens
# to gather contextual information.
#
# ------------------------------------------------------------
# IMPORTANT IDEA
# ------------------------------------------------------------
#
# Attention creates:
#
#     dynamic interaction topology.
#
# This means:
#
#     information flow changes continuously
#     depending on context.
#
# ------------------------------------------------------------
# DEEP QUESTION
# ------------------------------------------------------------
#
# Could meaning itself emerge through:
#
#     interaction patterns between representations?
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
#     different tokens will attend
#     to different contextual regions.
#
# Expected observations:
#
#     • verbs attend to subjects
#     • pronouns attend to referenced nouns
#     • important semantic words receive stronger attention
#
# ------------------------------------------------------------
# IMPORTANT IDEA
# ------------------------------------------------------------
#
# Attention is NOT fixed.
#
# It dynamically changes depending on:
#
#     • sentence structure
#     • semantic relationships
#     • contextual dependencies
#
# ------------------------------------------------------------
# TRANSFORMER INSIGHT
# ------------------------------------------------------------
#
# Meaning may emerge through:
#
#     contextual interaction dynamics.
#
# ------------------------------------------------------------
# THINK DEEPLY
# ------------------------------------------------------------
#
# Is attention:
#
#     • information routing?
#     • contextual retrieval?
#     • dynamic memory access?
#     • relational interaction?
#



#%% [markdown]
# ============================================================
# 3. LOAD REAL TRANSFORMER SYSTEM
# ============================================================
#
# We now load a REAL transformer model.
#
# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------
#
# We use:
#
#     BERT (Bidirectional Encoder Representations
#     from Transformers)
#
# BERT revolutionized NLP through:
#
#     contextual representation learning.
#
# ------------------------------------------------------------
# WHY BERT?
# ------------------------------------------------------------
#
# BERT provides:
#
#     • attention weights
#     • contextual embeddings
#     • multi-head attention
#     • layerwise representations
#
# making it ideal for representation analysis.
#
# ------------------------------------------------------------
# GLOSSARY OF FUNCTIONS
# ------------------------------------------------------------
#
# BertTokenizer.from_pretrained(...)
#     Loads pretrained tokenizer.
#
# BertModel.from_pretrained(...)
#     Loads pretrained transformer.
#
# tokenizer(...)
#     Converts text into token IDs.
#
# output_attentions=True
#     Returns attention matrices.
#
# softmax(...)
#     Converts scores into probabilities.
#
# attention_weights
#     Token-to-token interaction strengths.
#
# ------------------------------------------------------------
# IMPORTANT INSIGHT
# ------------------------------------------------------------
#
# Transformers do not process words independently.
#
# Instead:
#
#     representations continuously interact.
#



#%%

import torch

from transformers import (
    BertTokenizer,
    BertModel
)

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased"
)

# Load transformer model
model = BertModel.from_pretrained(
    "bert-base-uncased",
    output_attentions=True
)

print("Transformer system loaded successfully.")



#%% [markdown]
# ============================================================
# 4. TOKENIZATION AND INPUT PROCESSING
# ============================================================
#
# Transformers do NOT directly process raw text.
#
# Instead:
#
#     text -> tokens -> token IDs -> embeddings
#
# ------------------------------------------------------------
# TOKENIZATION
# ------------------------------------------------------------
#
# Example:
#
#     "Transformers analyze context"
#
# may become:
#
#     ["transformers", "analyze", "context"]
#
# or even:
#
#     subword fragments.
#
# ------------------------------------------------------------
# IMPORTANT IDEA
# ------------------------------------------------------------
#
# Attention operates at:
#
#     token level.
#
# ------------------------------------------------------------
# QUESTION
# ------------------------------------------------------------
#
# Which tokens will interact strongly?
#



#%%

sentence = (
    "The scientist analyzed the transformer attention map carefully."
)

# Tokenize sentence
inputs = tokenizer(
    sentence,
    return_tensors="pt"
)

print("Input IDs shape:")
print(inputs["input_ids"].shape)



#%%

# Convert token IDs back into readable tokens

tokens = tokenizer.convert_ids_to_tokens(
    inputs["input_ids"][0]
)

print("\nTOKENS\n")

print(tokens)



#%% [markdown]
# ============================================================
# 5. ATTENTION MECHANISM MATHEMATICS
# ============================================================
#
# CORE IDEA
# ---------
#
# Attention computes:
#
#     how strongly tokens interact.
#
# ------------------------------------------------------------
# QUERY, KEY, VALUE
# ------------------------------------------------------------
#
# Each token generates:
#
#     • Query vector (Q)
#     • Key vector (K)
#     • Value vector (V)
#
# ------------------------------------------------------------
# INTUITION
# ------------------------------------------------------------
#
# Query:
#
#     "What information am I looking for?"
#
# Key:
#
#     "What information do I contain?"
#
# Value:
#
#     "What information should I provide?"
#
# ============================================================
# ATTENTION FORMULA
# ============================================================
#
# Scaled dot-product attention:
#



#
# where:
#
#     Q = query matrix
#     K = key matrix
#     V = value matrix
#     d_k = key dimension
#
# ------------------------------------------------------------
# STEP-BY-STEP INTERPRETATION
# ------------------------------------------------------------
#
# Step 1:
#
#     Compute similarity:
#
#         QKᵀ
#
# This measures:
#
#     token interaction strength.
#
# ------------------------------------------------------------
# Step 2:
#
#     Scale by:
#
#         √d_k
#
# to stabilize gradients.
#
# ------------------------------------------------------------
# Step 3:
#
#     Apply softmax
#
# converting scores into:
#
#     probability distribution.
#
# ------------------------------------------------------------
# Step 4:
#
#     Weighted combination of values.
#
# This creates:
#
#     context-aware representations.
#
# ------------------------------------------------------------
# IMPORTANT INSIGHT
# ------------------------------------------------------------
#
# Attention dynamically redistributes:
#
#     information flow across tokens.
#
# ------------------------------------------------------------
# DEEP QUESTION
# ------------------------------------------------------------
#
# Is attention fundamentally:
#
#     dynamic contextual routing?
#



#%% [markdown]
# ============================================================
# 6. EXTRACT ATTENTION MAPS
# ============================================================
#
# We now pass text through the transformer.
#
# The model returns:
#
#     • hidden states
#     • attention weights
#
# ------------------------------------------------------------
# ATTENTION TENSOR STRUCTURE
# ------------------------------------------------------------
#
# Attention tensor dimensions:
#
#     [batch, heads, tokens, tokens]
#
# Example:
#
#     [1, 12, 10, 10]
#
# meaning:
#
#     • 1 sentence
#     • 12 attention heads
#     • 10 query tokens
#     • 10 key tokens
#
# ------------------------------------------------------------
# IMPORTANT IDEA
# ------------------------------------------------------------
#
# Each attention head learns:
#
#     different interaction patterns.
#



#%%

# Forward pass through transformer

with torch.no_grad():

    outputs = model(**inputs)

# Extract attention tensors
attentions = outputs.attentions

print("Number of transformer layers:")
print(len(attentions))

print("\nAttention tensor shape:")
print(attentions[0].shape)



#%% [markdown]
# ============================================================
# 7. ATTENTION MAP VISUALIZATION
# ============================================================
#
# Human beings cannot directly interpret:
#
#     high-dimensional attention tensors.
#
# So we visualize:
#
#     attention heatmaps.
#
# ------------------------------------------------------------
# HEATMAP INTERPRETATION
# ------------------------------------------------------------
#
# Rows:
#
#     query tokens
#
# Columns:
#
#     attended tokens
#
# Bright regions:
#
#     stronger attention interactions.
#
# ------------------------------------------------------------
# IMPORTANT QUESTION
# ------------------------------------------------------------
#
# Which tokens dominate information flow?
#



#%%

import matplotlib.pyplot as plt
import seaborn as sns

# Select:
# Layer 0
# Head 0

attention_map = attentions[0][0][0].cpu().numpy()

print("Attention map shape:")
print(attention_map.shape)



#%%

plt.figure(figsize=(12, 10))

sns.heatmap(
    attention_map,
    xticklabels=tokens,
    yticklabels=tokens,
    cmap="viridis",
    annot=False
)

plt.title(
    "Transformer Attention Map\n"
    "Layer 0 — Head 0"
)

plt.xlabel("Attended Tokens")
plt.ylabel("Query Tokens")

plt.show()



#%% [markdown]
# ============================================================
# 8. ATTENTION DISTRIBUTION ANALYSIS
# ============================================================
#
# We now inspect:
#
#     how strongly each token distributes attention.
#
# ------------------------------------------------------------
# IMPORTANT IDEA
# ------------------------------------------------------------
#
# Attention distributions reveal:
#
#     contextual dependency structure.
#
# ------------------------------------------------------------
# QUESTION
# ------------------------------------------------------------
#
# Which tokens become:
#
#     information hubs?
#



#%%

query_index = 3

query_token = tokens[query_index]

attention_distribution = attention_map[query_index]

print(f"\nATTENTION DISTRIBUTION FOR TOKEN: {query_token}\n")

for token, score in zip(tokens, attention_distribution):

    print(f"{token:15s} Attention: {score:.4f}")



#%% [markdown]
# ============================================================
# 9. INTERPRETATION
# ============================================================
#
# OBSERVATION ANALYSIS
# --------------------
#
# Examine the attention map carefully.
#
# Possible observations:
#
#     • nearby words interact strongly
#     • important nouns attract attention
#     • verbs attend to subjects
#     • punctuation influences structure
#
# ------------------------------------------------------------
# IMPORTANT REALIZATION
# ------------------------------------------------------------
#
# Transformers continuously:
#
#     redistribute information flow.
#
# This creates:
#
#     dynamic contextual representations.
#
# ------------------------------------------------------------
# ATTENTION AS TOPOLOGY
# ------------------------------------------------------------
#
# Attention maps define:
#
#     interaction topology between tokens.
#
# This topology changes:
#
#     sentence-by-sentence.
#
# ------------------------------------------------------------
# VERY IMPORTANT INSIGHT
# ------------------------------------------------------------
#
# Meaning may emerge through:
#
#     contextual interaction patterns
#
# rather than isolated representations.
#
# ------------------------------------------------------------
# DEEP QUESTION
# ------------------------------------------------------------
#
# Could cognition itself emerge through:
#
#     dynamic routing of information?
#



#%% [markdown]
# ============================================================
# 10. FAILURE CASES
# ============================================================
#
# REAL SYSTEMS ARE IMPERFECT
# --------------------------
#
# Attention analysis possesses limitations.
#
# ------------------------------------------------------------
# LIMITATION 1 — ATTENTION IS NOT EXPLANATION
# ------------------------------------------------------------
#
# High attention does NOT necessarily imply:
#
#     causal importance.
#
# Attention reflects:
#
#     interaction weighting
#
# not guaranteed reasoning.
#
# ------------------------------------------------------------
# LIMITATION 2 — MULTI-HEAD COMPLEXITY
# ------------------------------------------------------------
#
# Different heads learn:
#
#     different relational structures.
#
# One heatmap captures only:
#
#     partial system behavior.
#
# ------------------------------------------------------------
# LIMITATION 3 — CONTEXT DEPENDENCE
# ------------------------------------------------------------
#
# Attention changes dynamically
# across different sentences.
#
# ------------------------------------------------------------
# LIMITATION 4 — VISUALIZATION DISTORTION
# ------------------------------------------------------------
#
# Heatmaps simplify:
#
#     extremely high-dimensional interactions.
#
# ------------------------------------------------------------
# THINK CRITICALLY
# ------------------------------------------------------------
#
# Is attention sufficient to explain:
#
#     transformer reasoning?
#



#%% [markdown]
# ============================================================
# 11. SCIENTIFIC INSIGHT
# ============================================================
#
# MOST IMPORTANT INSIGHT
# ----------------------
#
# Transformers operate through:
#
#     dynamic representation interaction.
#
# ------------------------------------------------------------
# ATTENTION ENABLES
# ------------------------------------------------------------
#
#     • contextual reasoning
#     • semantic disambiguation
#     • long-range dependency modeling
#     • information routing
#     • adaptive contextualization
#
# ------------------------------------------------------------
# FOUNDATIONAL REALIZATION
# ------------------------------------------------------------
#
# Meaning is NOT static.
#
# It continuously evolves through:
#
#     contextual interaction dynamics.
#
# ------------------------------------------------------------
# IMPORTANT TRANSITION
# ------------------------------------------------------------
#
# We are no longer studying:
#
#     isolated embedding geometry
#
# but:
#
#     dynamic information flow systems.
#
# ------------------------------------------------------------
# LOOK AHEAD
# ------------------------------------------------------------
#
# Next steps investigate:
#
#     • token interaction analysis
#     • head specialization
#     • contextual representation shift
#     • layerwise representation evolution
#
# We are progressively uncovering:
#
#     AI as adaptive representation routing.
#



#%% [markdown]
# ============================================================
# 12. EXPLORATION EXERCISES
# ============================================================
#
# EXPLORATION 1
# -------------
#
# Try ambiguous sentences:
#
#     "The bank approved the loan."
#     "The boat reached the bank."
#
# How does attention change?
#
# ------------------------------------------------------------
# EXPLORATION 2
# -------------
#
# Compare short and long sentences.
#
# Does attention become:
#
#     more distributed?
#
# ------------------------------------------------------------
# EXPLORATION 3
# -------------
#
# Visualize:
#
#     different transformer layers.
#
# Do deeper layers learn:
#
#     different interaction structures?
#
# ------------------------------------------------------------
# EXPLORATION 4
# -------------
#
# Compare:
#
#     different attention heads.
#
# Do some heads specialize in:
#
#     syntax?
#     semantics?
#     positional structure?
#
# ------------------------------------------------------------
# EXPLORATION 5
# -------------
#
# Replace BERT with:
#
#     • DistilBERT
#     • RoBERTa
#     • GPT-2
#
# How does attention topology change?
#
# ------------------------------------------------------------
# FINAL QUESTION
# ------------------------------------------------------------
#
# If modern AI systems fundamentally operate through:
#
#     dynamic information routing,
#
# then could intelligence itself emerge from:
#
#     adaptive interaction topology?
#
# ============================================================