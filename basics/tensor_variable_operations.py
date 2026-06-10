
# %% [Markdown]
# # PyTorch Variable Operations Playground
#
# In this notebook/script we explore:
#
# - tensor.requires_grad_()
# - tensor.detach()
# - tensor.clone()
#
# These operations are deeply connected to:
#
# - autograd
# - computational graphs
# - gradient flow
# - memory management
# - tensor tracking
#
# They are extremely important in:
#
# - neural networks
# - reinforcement learning
# - transformers
# - diffusion models
# - latent semantic systems

# %%
import torch

BREAK_SIZE = 80


# %% [Markdown]
# # Base Tensor
#
# We first create a tensor WITHOUT gradient tracking.

# %%
x = torch.tensor([1.0, 2.0, 3.0])

print("=" * BREAK_SIZE)
print("BASE TENSOR")
print("=" * BREAK_SIZE)

print(x)

print("\nrequires_grad :")
print(x.requires_grad)


# %% [Markdown]
# # tensor.requires_grad_()
#
# ## DESCRIPTION
#
# Enables gradient tracking IN-PLACE.
#
# The underscore `_` means:
#
# > modify the existing tensor directly.
#
# ---
#
# ## MATHEMATICAL IDEA
#
# Suppose:
#
# y = f(x)
#
# We want:
#
# dy/dx
#
# Then:
#
# x must participate in the computational graph.
#
# requires_grad=True tells PyTorch:
#
# > "Track operations on this tensor."
#
# ---
#
# ## BASIC ALGORITHM
#
# 1. Mark tensor as differentiable
# 2. Record future operations
# 3. Build computational graph
# 4. Enable backpropagation

# %%
print("\n" + "#" * BREAK_SIZE)
print("method tensor.requires_grad_()")

x.requires_grad_()

print("\nTensor x :")
print(x)

print("\nrequires_grad :")
print(x.requires_grad)


# %% [Markdown]
# # Build Computational Graph
#
# ## Mathematical Function
#
# y = x² + 2x
#
# ## Derivative
#
# dy/dx = 2x + 2

# %%
y = x**2 + 2*x

print("\nTensor y :")
print(y)

print("\nGradient function :")
print(y.grad_fn)


# %% [Markdown]
# # Backpropagation
#
# We now compute:
#
# dy/dx

# %%
print("\n" + "#" * BREAK_SIZE)
print("BACKPROPAGATION")

loss = y.sum()

loss.backward()

print("\nGradient of x :")
print(x.grad)


# %% [Markdown]
# # Mathematical Verification
#
# For:
#
# y = x² + 2x
#
# dy/dx = 2x + 2
#
# If:
#
# x = [1, 2, 3]
#
# Then:
#
# dy/dx =
#
# [4, 6, 8]

# %% [Markdown]
# # tensor.detach()
#
# ## DESCRIPTION
#
# Creates tensor detached from computational graph.
#
# Detached tensor:
#
# - shares same memory
# - does NOT track gradients
# - breaks gradient propagation
#
# ---
#
# ## MATHEMATICAL IDEA
#
# Suppose:
#
# z = g(y)
#
# Normally:
#
# dz/dx propagates through y
#
# But detach():
#
# cuts graph connection
#
# So:
#
# dz/dx stops propagating.
#
# ---
#
# ## BASIC ALGORITHM
#
# 1. Create view of tensor
# 2. Remove graph tracking
# 3. Preserve underlying data
# 4. Prevent gradient flow

# %%
print("\n" + "#" * BREAK_SIZE)
print("method tensor.detach()")

a = torch.tensor(
    [1.0, 2.0, 3.0],
    requires_grad=True
)

b = a * 2

print("\nTensor b :")
print(b)

print("\nGradient tracking :")
print(b.requires_grad)


# %%
c = b.detach()

print("\nDetached tensor c :")
print(c)

print("\nGradient tracking :")
print(c.requires_grad)


# %% [Markdown]
# # Important Observation
#
# c:
#
# - still contains same values
# - but no longer participates in autograd

# %%
print("\nGradient function of b :")
print(b.grad_fn)

print("\nGradient function of c :")
print(c.grad_fn)


# %% [Markdown]
# # Memory Sharing Demonstration
#
# detach() shares memory with original tensor.

# %%
print("\n" + "#" * BREAK_SIZE)
print("MEMORY SHARING DEMONSTRATION")

print("\nOriginal c :")
print(c)

b[0] = 999

print("\nModified b :")
print(b)

print("\nDetached c after modifying b :")
print(c)


# %% [Markdown]
# # Important Insight
#
# detach() does NOT copy memory.
#
# It only disconnects autograd graph.

# %% [Markdown]
# # tensor.clone()
#
# ## DESCRIPTION
#
# Creates COPY of tensor.
#
# Unlike detach():
#
# clone():
#
# - allocates new memory
# - copies values
#
# ---
#
# ## MATHEMATICAL IDEA
#
# clone():
#
# creates independent tensor:
#
# x_clone = x
#
# but physically stored separately.
#
# ---
#
# ## BASIC ALGORITHM
#
# 1. Allocate new memory
# 2. Copy tensor values
# 3. Preserve tensor structure
# 4. Return independent tensor

# %%
print("\n" + "#" * BREAK_SIZE)
print("method tensor.clone()")

p = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0]
])

print("\nOriginal tensor p :")
print(p)

q = p.clone()

print("\nCloned tensor q :")
print(q)


# %% [Markdown]
# # Independent Memory Demonstration
#
# clone() creates separate memory.

# %%
p[0][0] = 999

print("\nModified p :")
print(p)

print("\nCloned q after modifying p :")
print(q)


# %% [Markdown]
# # Important Observation
#
# q did NOT change.
#
# Therefore:
#
# clone() copied memory.

# %% [Markdown]
# # clone() + detach()
#
# This combination is EXTREMELY common.
#
# Why?
#
# detach():
# - removes graph tracking
#
# clone():
# - creates independent memory
#
# Together:
#
# safe independent tensor copy.

# %%
print("\n" + "#" * BREAK_SIZE)
print("clone() + detach()")

u = torch.tensor(
    [1.0, 2.0, 3.0],
    requires_grad=True
)

v = u.clone().detach()

print("\nTensor u :")
print(u)

print("\nTensor v :")
print(v)

print("\nv requires_grad :")
print(v.requires_grad)


# %% [Markdown]
# # Why This Pattern Matters
#
# Frequently used in:
#
# - target networks
# - reinforcement learning
# - inference pipelines
# - tensor snapshots
# - stable latent representations

# %% [Markdown]
# # FINAL SUMMARY

# %%
print("\n" + "=" * BREAK_SIZE)
print("VARIABLE OPERATIONS SUMMARY")
print("=" * BREAK_SIZE)

print("""
1. requires_grad_()
   -> enables gradient tracking

2. detach()
   -> disconnects tensor from graph
   -> shares memory

3. clone()
   -> creates independent memory copy

Important Deep Learning Uses:

- autograd control
- memory management
- stable inference
- reinforcement learning
- transformer pipelines
- latent state preservation
""")

print("=" * BREAK_SIZE)