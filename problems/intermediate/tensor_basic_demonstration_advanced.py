# %% [Markdown]
# # PyTorch Problem — 3D Gaussian Averaging Filter
#
# ## Problem Statement
#
# Generate random tensors of size:
#
# 28 × 28
#
# with:
#
# 12 batches
#
# and:
#
# 3 channels
#
# You can think of these as:
#
# RGB image-like tensors.
#
# Tensor Shape:
#
# (Batch, Channel, Height, Width)
#
# =
#
# (12, 3, 28, 28)
#
# ---
#
# ## Objective
#
# Apply:
#
# separable 3D Gaussian averaging filter
#
# in:
#
# - X direction
# - Y direction
# - Z direction (channel direction)
#
# Since Gaussian filters are linearly separable:
#
# instead of one giant 3D convolution,
#
# we apply:
#
# - separate X filter
# - separate Y filter
# - separate Z filter
#
# sequentially.
#
# ---
#
# ## Constraint
#
# Filter length must be:
#
# less than half of dimension size.
#
# Since:
#
# dimension = 28
#
# therefore:
#
# filter size < 14
#
# We choose:
#
# kernel_size = 7
#
# ---
#
# ## Mathematical Idea
#
# A Gaussian function:
#
# G(x)=e^{-x^2/2\sigma^2}
#
# Normalized Gaussian:
#
# G_n(x)=\frac{G(x)}{\sum_i G(x_i)}
#
# ---
#
# ## Separable Filtering
#
# Instead of:
#
# G(x,y,z)
#
# we use:
#
# G(x)G(y)G(z)
#
# This drastically reduces computation.
#
# Complexity reduces from:
#
# O(k³)
#
# to:
#
# O(3k)
#
# which is one of the important ideas in image processing
# and deep learning systems.

# %%
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

BREAK_SIZE = 80


# %% [Markdown]
# # Create Random Tensor
#
# Shape:
#
# (12, 3, 28, 28)
#
# Meaning:
#
# - 12 batches
# - 3 channels
# - 28 height
# - 28 width

# %%
print("=" * BREAK_SIZE)
print("GENERATING RANDOM TENSOR")
print("=" * BREAK_SIZE)

x = torch.rand(12, 3, 28, 28)

print("Tensor Shape :")
print(x.shape)


# %% [Markdown]
# # Create 1D Gaussian Kernel
#
# ## Mathematical Formula
#
# G(x)=e^{-x^2/2\sigma^2}
#
# We then normalize:
#
# G_n(x)=\frac{G(x)}{\sum_i G(x_i)}

# %%
def gaussian_kernel_1d(kernel_size=7, sigma=1.5):

    ####################################################################
    # Create centered coordinate system
    ####################################################################

    coords = torch.arange(kernel_size) - kernel_size // 2

    ####################################################################
    # Gaussian Formula
    ####################################################################

    kernel = torch.exp(-(coords**2) / (2 * sigma**2))

    ####################################################################
    # Normalize kernel
    ####################################################################

    kernel = kernel / kernel.sum()

    return kernel


# %%
print("=" * BREAK_SIZE)
print("GAUSSIAN KERNEL")
print("=" * BREAK_SIZE)

kernel_1d = gaussian_kernel_1d()

print(kernel_1d)

print("\nKernel Sum :")
print(kernel_1d.sum())


# %% [Markdown]
# # Why Normalize?
#
# Without normalization:
#
# brightness/intensity may increase or decrease.
#
# Normalization ensures:
#
# total energy preserved.

# %% [Markdown]
# # Create X-Direction Filter
#
# Shape needed for conv2d:
#
# (out_channels, in_channels/groups, kernel_height, kernel_width)

# %%
kernel_x = kernel_1d.view(1, 1, 1, -1)

print(kernel_x.shape)


# %% [Markdown]
# # Create Y-Direction Filter

# %%
kernel_y = kernel_1d.view(1, 1, -1, 1)

print(kernel_y.shape)


# %% [Markdown]
# # Create Z-Direction Filter
#
# Channel-direction filtering.
#
# We reshape for channel mixing.

# %%
kernel_z = kernel_1d[:3]

kernel_z = kernel_z / kernel_z.sum()

print(kernel_z)


# %% [Markdown]
# # Apply X-Direction Gaussian Filtering
#
# ## Mathematical Idea
#
# Convolution:
#
# y(i,j)=\sum_k x(i,j-k)G(k)

# %%
print("=" * BREAK_SIZE)
print("APPLYING X FILTER")
print("=" * BREAK_SIZE)

filtered_x = F.conv2d(
    x,
    kernel_x.repeat(3, 1, 1, 1),
    padding=(0, 3),
    groups=3
)

print(filtered_x.shape)


# %% [Markdown]
# # Why groups=3?
#
# Each channel:
#
# - R
# - G
# - B
#
# is filtered independently.

# %% [Markdown]
# # Apply Y-Direction Gaussian Filtering
#
# ## Mathematical Idea
#
# y(i,j)=\sum_k x(i-k,j)G(k)

# %%
print("=" * BREAK_SIZE)
print("APPLYING Y FILTER")
print("=" * BREAK_SIZE)

filtered_xy = F.conv2d(
    filtered_x,
    kernel_y.repeat(3, 1, 1, 1),
    padding=(3, 0),
    groups=3
)

print(filtered_xy.shape)


# %% [Markdown]
# # Apply Z-Direction Filtering
#
# We now smooth across channels.
#
# This behaves somewhat like:
#
# inter-channel Gaussian averaging.

# %%
print("=" * BREAK_SIZE)
print("APPLYING Z FILTER")
print("=" * BREAK_SIZE)

filtered_xyz = filtered_xy.clone()

for batch in range(filtered_xy.shape[0]):

    for i in range(filtered_xy.shape[2]):

        for j in range(filtered_xy.shape[3]):

            pixel_vector = filtered_xy[batch, :, i, j]

            filtered_xyz[batch, :, i, j] = (
                pixel_vector * kernel_z
            ).sum()

print(filtered_xyz.shape)


# %% [Markdown]
# # Final Output Shape
#
# Shape preserved:
#
# (12, 3, 28, 28)

# %%
print("=" * BREAK_SIZE)
print("FINAL FILTERED OUTPUT")
print("=" * BREAK_SIZE)

print(filtered_xyz.shape)


# %% [Markdown]
# # Visualization
#
# Show:
#
# first batch
#
# first 3 channels
#
# after filtering.

# %%
print("=" * BREAK_SIZE)
print("VISUALIZATION")
print("=" * BREAK_SIZE)

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for channel in range(3):

    axes[channel].imshow(
        filtered_xyz[0, channel].detach().numpy(),
        cmap="gray"
    )

    axes[channel].set_title(
        f"Filtered Channel {channel}"
    )

    axes[channel].axis("off")

plt.tight_layout()

plt.show()


# %% [Markdown]
# # Important Deep Learning Insights
#
# This small exercise demonstrates:
#
# - tensor generation
# - convolution
# - Gaussian filtering
# - tensor reshaping
# - grouped convolution
# - channel-wise processing
# - separable filters
# - multidimensional tensor operations
#
# ---
#
# ## Why This Matters
#
# Similar operations appear in:
#
# - CNNs
# - image preprocessing
# - diffusion models
# - transformer embeddings
# - latent space smoothing
# - feature extraction
# - volumetric data processing
#
# ---
#
# ## Computational Insight
#
# Separable Gaussian filters are important because:
#
# 3D convolution:
#
# O(k³)
#
# becomes:
#
# O(3k)
#
# This dramatically reduces:
#
# - memory
# - computation
# - kernel complexity
#
# which is a fundamental optimization idea in
# computer vision and deep learning.