# %% [Markdown]
# Problem Statement — Student Marks Tensor Analysis
#
# You are given marks of students in three subjects:
#
# Mathematics
# Physics
# Chemistry
#
# The marks are represented using a tensor where:
#
# each row represents one student
# each column represents one subject
#
# The tensor is given below:
#
# marks = torch.tensor([
#     [80, 70, 90],
#     [60, 75, 85],
#     [95, 88, 92],
#     [50, 65, 70]
# ]).float()

# %% [Markdown]
# Tasks

# %%
# marks = torch.tensor([
#     [80, 70, 90],
#     [60, 75, 85],
#     [95, 88, 92],
#     [50, 65, 70]
# ]).float()
# Initialize your tensor here

# %% [Markdown]


# ##  Task 1 — Analyze Tensor Information
#
# Print the following properties of the tensor:
#
# shape
# number of dimensions
# total number of elements
# datatype

# %%
# Space to compute Task 1

# %% [Markdown]
# ##  Task 2 — Compute Total Marks Per Student
#
# Using tensor reduction operations, compute:
#
# total marks obtained by each student.
# Hint:
# Use dim in the argument

# %%
# Space to compute Task 2

# %% [Markdown]
# ##  Task 3 — Compute Average Marks Per Student
#
# Using tensor arithmetic operations, compute:
#
# average marks obtained by each student.
#
# Use:
#
# total marks
# division operation

# %%
# Space to compute Task 3

# %% [Markdown]
# ##  Task 4 — Compute Subject-wise Average
#
# Find the average marks for:
#
# Mathematics
# Physics
# Chemistry
#
# using reduction operations.
# %%
# Space to compute Task 4

# %% [Markdown]
# ##  Task 5 — Add Batch Dimension
#
# Suppose the tensor now needs to be passed into a neural-network-like system that expects a batch dimension.
#
# Convert tensor shape from:
#
# (4, 3)
#
# to:
#
# (1, 4, 3)
#
# using:
#
# unsqueeze()
# Expected Shape After Unsqueeze:
# (1, 4, 3)

# %%
# Space to compute Task 5

# %% [Markdown]
# ##  Task 6 — Remove Batch Dimension
#
# Remove the added batch dimension using:
#
# squeeze()

# %%
# Space to compute Task 6


# %% [Markdown]
# ##  Task 7 — Find Maximum and Minimum Marks
#
# Find:
#
# highest mark in the tensor
# lowest mark in the tensor
# index position of highest mark
# index position of lowest mark
# %%
# Space to compute Task 7

