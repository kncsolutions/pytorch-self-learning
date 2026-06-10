
# %% [markdown]
"""
# PLAY WITH KALU
# STEP 03 — TENSOR PREPROCESSING

## CORE INTUITION

Deep learning systems require geometric consistency.

Raw images may have:

- different sizes
- different pixel distributions
- different lighting conditions
- different orientations

Preprocessing transforms raw images into stable tensor structures
suitable for learning systems.

This step introduces:

- resizing
- normalization
- tensor standardization
- preprocessing pipelines
- augmentation intuition
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
"""
# %%


# ============================================================
# LOAD IMAGE
# ============================================================

image = Image.open(IMAGE_PATH).convert("RGB")

print("\n[ORIGINAL IMAGE INFORMATION]")
print("-" * 50)

print(f"Image Size : {image.size}")
print(f"Image Mode : {image.mode}")


# %% [markdown]
"""
# BASIC TENSOR CONVERSION

First convert the image into a tensor without preprocessing.

This helps us compare:

- raw tensor structure
- preprocessed tensor structure
"""
# %%


# ============================================================
# RAW IMAGE → TENSOR
# ============================================================

basic_transform = transforms.ToTensor()

raw_tensor = basic_transform(image)

print("\n[RAW TENSOR INFORMATION]")
print("-" * 50)

print(f"Tensor Shape : {raw_tensor.shape}")
print(f"Tensor Mean  : {raw_tensor.mean():.4f}")
print(f"Tensor Std   : {raw_tensor.std():.4f}")


# %% [markdown]
"""
# RESIZING

Neural networks usually expect fixed tensor dimensions.

Example:

    [3, 224, 224]

Why resizing matters:

- batch consistency
- computational efficiency
- stable tensor geometry
- fixed network input dimensions
"""
# %%


# ============================================================
# IMAGE RESIZING
# ============================================================

resize_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ]
)

resized_tensor = resize_transform(image)

print("\n[RESIZED TENSOR INFORMATION]")
print("-" * 50)

print(f"Tensor Shape : {resized_tensor.shape}")


# %% [markdown]
"""
# NORMALIZATION

Normalization stabilizes numerical distributions.

Without normalization:

    tensor values may vary significantly

With normalization:

    tensor distributions become more stable

Common normalization formula:

                x - mean
    normalized = ----------
                   std

This improves:

- optimization stability
- gradient flow
- convergence behavior
"""
# %%


# ============================================================
# NORMALIZATION
# ============================================================

normalized_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ]
)

normalized_tensor = normalized_transform(image)

print("\n[NORMALIZED TENSOR INFORMATION]")
print("-" * 50)

print(f"Tensor Shape : {normalized_tensor.shape}")

print(
    f"Minimum Value : "
    f"{normalized_tensor.min():.4f}"
)

print(
    f"Maximum Value : "
    f"{normalized_tensor.max():.4f}"
)

print(
    f"Tensor Mean   : "
    f"{normalized_tensor.mean():.4f}"
)

print(
    f"Tensor Std    : "
    f"{normalized_tensor.std():.4f}"
)


# %% [markdown]
"""
# PREPROCESSING PIPELINE

Deep learning systems usually apply multiple transformations
through a preprocessing pipeline.

Pipeline Thinking:

    Raw Image
        ↓
    Resize
        ↓
    Tensor Conversion
        ↓
    Normalization
        ↓
    Learning System
"""
# %%


# ============================================================
# FULL PREPROCESSING PIPELINE
# ============================================================

preprocessing_pipeline = transforms.Compose(
    [
        transforms.Resize((224, 224)),

        transforms.RandomHorizontalFlip(
            p=0.5
        ),

        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ]
)

processed_tensor = preprocessing_pipeline(image)

print("\n[FINAL PREPROCESSED TENSOR]")
print("-" * 50)

print(f"Tensor Shape : {processed_tensor.shape}")

print(f"Tensor Dtype : {processed_tensor.dtype}")


# %% [markdown]
"""
# AUGMENTATION INTUITION

Augmentation artificially creates variation.

This helps the network learn:

- robustness
- invariance
- generalized representations

Examples:

- flipping
- rotation
- brightness adjustment
- cropping

The goal is not memorization.

The goal is representation learning.
"""
# %%


# ============================================================
# VISUALIZATION PREPARATION
# ============================================================

visual_tensor = normalized_tensor.clone()

visual_tensor = (
    visual_tensor * 0.5
) + 0.5

visual_image = visual_tensor.permute(1, 2, 0)


# %% [markdown]
"""
# VISUALIZATION

We visualize:

- original image
- normalized image

to understand preprocessing effects.
"""
# %%


# ============================================================
# VISUALIZATION
# ============================================================

figure, axis = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(12, 6)
)

axis[0].imshow(image)

axis[0].set_title("Original Image")

axis[0].axis("off")

axis[1].imshow(visual_image)

axis[1].set_title("Preprocessed Tensor")

axis[1].axis("off")

plt.tight_layout()

plt.show()


# %% [markdown]
"""
# MEMORY INTUITION

Preprocessing itself is also tensor transformation.

Each transformation modifies:

- tensor geometry
- numerical distribution
- representation structure

Deep learning pipelines are ultimately chains of tensor operations.
"""
# %%


# ============================================================
# MEMORY INFORMATION
# ============================================================

num_elements = processed_tensor.numel()

memory_bytes = (
    num_elements *
    processed_tensor.element_size()
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

Preprocessing transforms:

    raw visual data
        ↓
    stable tensor geometry
        ↓
    learnable numerical structure

This is one of the foundational stages of machine perception.
"""
# %%