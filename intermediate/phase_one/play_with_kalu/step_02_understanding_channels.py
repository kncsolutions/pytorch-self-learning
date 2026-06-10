
# %% [markdown]
"""
# PLAY WITH KALU
# STEP 02 — UNDERSTANDING CHANNELS

## CORE INTUITION

An RGB image is not a single entity.

It is composed of multiple information fields:

- Red Channel
- Green Channel
- Blue Channel

PyTorch represents these channels as stacked tensor planes.

Tensor Shape:

    [Channels, Height, Width]

Example:

    [3, 224, 224]

Where:

    3     -> RGB channels
    224   -> Height
    224   -> Width

Deep learning systems initially do not see:

- dog
- eyes
- fur
- ears

They first observe:

- intensity distributions
- spatial numerical patterns
- structured tensor fields
"""
# %%


# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path

from PIL import Image

import matplotlib.pyplot as plt

import torch
from torchvision import transforms


# %% [markdown]
"""
# IMAGE CONFIGURATION
"""
# %%


# ============================================================
# IMAGE PATH
# ============================================================

IMAGE_PATH = Path("dataset/dogs/d1.jpg")


# %% [markdown]
"""
# LOAD IMAGE

PIL loads the image in RGB format.

Typical Image Shape (PIL Perspective):

    [Height, Width, Channels]

PyTorch Perspective After Tensor Conversion:

    [Channels, Height, Width]
"""
# %%


# ============================================================
# LOAD IMAGE
# ============================================================

image = Image.open(IMAGE_PATH).convert("RGB")

print("\n[IMAGE INFORMATION]")
print("-" * 50)

print(f"Image Type : {type(image)}")
print(f"Image Mode : {image.mode}")
print(f"Image Size : {image.size}")


# %% [markdown]
"""
# IMAGE → TENSOR TRANSFORMATION

`ToTensor()` performs:

1. Numerical conversion
2. Dimension rearrangement
3. Pixel normalization

Pixel Value Conversion:

    [0, 255]
        ↓
    [0.0, 1.0]

Dimension Conversion:

    [H, W, C]
        ↓
    [C, H, W]
"""
# %%


# ============================================================
# IMAGE TO TENSOR
# ============================================================

transform = transforms.ToTensor()

image_tensor = transform(image)

print("\n[TENSOR INFORMATION]")
print("-" * 50)

print(f"Tensor Shape : {image_tensor.shape}")
print(f"Tensor Dtype : {image_tensor.dtype}")


# %% [markdown]
"""
# TENSOR GEOMETRY

Each channel is a separate tensor plane.

Channel Index Mapping:

    0 -> Red
    1 -> Green
    2 -> Blue
"""
# %%


# ============================================================
# TENSOR GEOMETRY
# ============================================================

channels, height, width = image_tensor.shape

print("\n[TENSOR GEOMETRY]")
print("-" * 50)

print(f"Channels : {channels}")
print(f"Height   : {height}")
print(f"Width    : {width}")


# %% [markdown]
"""
# CHANNEL EXTRACTION

We isolate each information field independently.

Each extracted channel becomes a 2D tensor:

    [Height, Width]
"""
# %%


# ============================================================
# CHANNEL EXTRACTION
# ============================================================

red_channel = image_tensor[0]
green_channel = image_tensor[1]
blue_channel = image_tensor[2]

print("\n[CHANNEL SHAPES]")
print("-" * 50)

print(f"Red Channel Shape   : {red_channel.shape}")
print(f"Green Channel Shape : {green_channel.shape}")
print(f"Blue Channel Shape  : {blue_channel.shape}")


# %% [markdown]
"""
# CHANNEL STATISTICS

Different channels may contain different statistical distributions.

These distributions become important during:

- normalization
- feature extraction
- convolution
- representation learning
"""
# %%


# ============================================================
# CHANNEL STATISTICS
# ============================================================

print("\n[CHANNEL STATISTICS]")
print("-" * 50)

print(
    f"Red Channel Mean   : {red_channel.mean():.4f}"
)

print(
    f"Green Channel Mean : {green_channel.mean():.4f}"
)

print(
    f"Blue Channel Mean  : {blue_channel.mean():.4f}"
)


# %% [markdown]
"""
# VISUALIZING CHANNELS

Each channel can be visualized independently.

This reveals how information is distributed across:

- color fields
- spatial regions
- intensity structures
"""
# %%


# ============================================================
# CHANNEL VISUALIZATION
# ============================================================

figure, axis = plt.subplots(
    nrows=1,
    ncols=3,
    figsize=(15, 5)
)

axis[0].imshow(
    red_channel,
    cmap="Reds"
)

axis[0].set_title("Red Channel")
axis[0].axis("off")

axis[1].imshow(
    green_channel,
    cmap="Greens"
)

axis[1].set_title("Green Channel")
axis[1].axis("off")

axis[2].imshow(
    blue_channel,
    cmap="Blues"
)

axis[2].set_title("Blue Channel")
axis[2].axis("off")

plt.tight_layout()

plt.show()


# %% [markdown]
"""
# FULL RGB RECONSTRUCTION

The original image emerges from the combination of:

- Red Tensor Plane
- Green Tensor Plane
- Blue Tensor Plane

This is tensor composition.
"""
# %%


# ============================================================
# RGB VISUALIZATION
# ============================================================

rgb_image = image_tensor.permute(1, 2, 0)

plt.figure(figsize=(6, 6))

plt.imshow(rgb_image)

plt.title("Reconstructed RGB Image")

plt.axis("off")

plt.show()


# %% [markdown]
"""
# MEMORY INTUITION

A tensor is structured numerical memory.

Image tensors store:

    Channels × Height × Width

numerical values inside contiguous computational structures.
"""
# %%


# ============================================================
# MEMORY INFORMATION
# ============================================================

num_elements = image_tensor.numel()

memory_bytes = (
    num_elements *
    image_tensor.element_size()
)

print("\n[MEMORY INFORMATION]")
print("-" * 50)

print(f"Tensor Elements : {num_elements}")

print(
    f"Approx Memory   : "
    f"{memory_bytes / 1024:.2f} KB"
)


# %% [markdown]
"""
# FINAL CONCEPTUAL SUMMARY

An RGB image inside PyTorch becomes:

    Image
        ↓
    Numerical Tensor
        ↓
    Channel Fields
        ↓
    Spatial Information System

Deep learning systems gradually learn patterns from these
structured tensor interactions.
"""