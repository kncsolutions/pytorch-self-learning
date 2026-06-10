
# %% [markdown]
"""
# PLAY WITH KALU
# STEP 05 — SIMPLE FEATURE LEARNING

## CORE INTUITION

Neural networks do not initially understand:

- dogs
- cats
- faces
- fur

They first learn simple visual structures such as:

- edges
- gradients
- textures
- intensity transitions

Convolution filters act as local pattern detectors.

This step introduces:

- convolution intuition
- feature extraction
- local spatial interaction
- activation maps
- learned visual patterns
"""
# %%


# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path

from PIL import Image

import matplotlib.pyplot as plt

import torch
import torch.nn as nn

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

The image is converted into a geometrically consistent tensor.
"""
# %%


# ============================================================
# IMAGE TRANSFORMATION PIPELINE
# ============================================================

transform_pipeline = transforms.Compose(
    [
        transforms.Resize((224, 224)),

        transforms.ToTensor()
    ]
)

image = Image.open(IMAGE_PATH).convert("RGB")

image_tensor = transform_pipeline(image)

print("\n[IMAGE TENSOR INFORMATION]")
print("-" * 50)

print(f"Tensor Shape : {image_tensor.shape}")


# %% [markdown]
"""
# BATCH DIMENSION

Convolution layers expect:

    [Batch, Channels, Height, Width]

Current tensor shape:

    [Channels, Height, Width]

We therefore introduce a batch dimension.
"""
# %%


# ============================================================
# ADD BATCH DIMENSION
# ============================================================

image_tensor = image_tensor.unsqueeze(0)

print("\n[BATCHED TENSOR INFORMATION]")
print("-" * 50)

print(f"Tensor Shape : {image_tensor.shape}")


# %% [markdown]
"""
# SIMPLE CONVOLUTION LAYER

A convolution layer learns local spatial patterns.

Example patterns:

- vertical edges
- horizontal edges
- texture structures
- contrast transitions

Core intuition:

Convolution =
localized information aggregation
"""
# %%


# ============================================================
# SIMPLE CONVOLUTION LAYER
# ============================================================

convolution_layer = nn.Conv2d(
    in_channels=3,

    out_channels=8,

    kernel_size=3,

    stride=1,

    padding=1
)

print("\n[CONVOLUTION LAYER]")
print("-" * 50)

print(convolution_layer)


# %% [markdown]
"""
# FEATURE EXTRACTION

The convolution layer transforms:

    raw image tensor
        ↓
    feature tensor

Output Shape:

    [Batch, Feature Maps, Height, Width]

Feature maps represent learned spatial responses.
"""
# %%


# ============================================================
# FEATURE EXTRACTION
# ============================================================

feature_tensor = convolution_layer(image_tensor)

print("\n[FEATURE TENSOR INFORMATION]")
print("-" * 50)

print(f"Feature Shape : {feature_tensor.shape}")


# %% [markdown]
"""
# FEATURE GEOMETRY

Feature Tensor Shape:

    [Batch, Feature Maps, Height, Width]

Example:

    [1, 8, 224, 224]

Interpretation:

    1 -> batch
    8 -> learned feature detectors
"""
# %%


# ============================================================
# FEATURE GEOMETRY
# ============================================================

batch_size, num_features, height, width = (
    feature_tensor.shape
)

print("\n[FEATURE GEOMETRY]")
print("-" * 50)

print(f"Batch Size  : {batch_size}")

print(f"Feature Maps: {num_features}")

print(f"Height      : {height}")

print(f"Width       : {width}")


# %% [markdown]
"""
# ACTIVATION FUNCTIONS

Activation functions introduce non-linearity.

Without activation:

    the network becomes a linear system

ReLU:

    max(0, x)

removes negative activations.
"""
# %%


# ============================================================
# RELU ACTIVATION
# ============================================================

relu = nn.ReLU()

activated_features = relu(feature_tensor)

print("\n[ACTIVATED FEATURES]")
print("-" * 50)

print(
    f"Minimum Value : "
    f"{activated_features.min():.4f}"
)

print(
    f"Maximum Value : "
    f"{activated_features.max():.4f}"
)


# %% [markdown]
"""
# FEATURE MAP VISUALIZATION

Each feature map represents a learned spatial response.

Different filters may respond to:

- edges
- fur textures
- brightness transitions
- structural patterns
"""
# %%


# ============================================================
# VISUALIZE FEATURE MAPS
# ============================================================

figure, axis = plt.subplots(
    nrows=2,
    ncols=4,
    figsize=(12, 6)
)

feature_maps = activated_features[0]

for index in range(8):

    row = index // 4

    column = index % 4

    axis[row][column].imshow(
        feature_maps[index].detach(),
        cmap="viridis"
    )

    axis[row][column].set_title(
        f"Feature {index + 1}"
    )

    axis[row][column].axis("off")

plt.tight_layout()

plt.show()


# %% [markdown]
"""
# FILTER WEIGHTS

The convolution layer contains learnable parameters.

Tensor Shape:

    [Out Channels, In Channels, Kernel Height, Kernel Width]

Example:

    [8, 3, 3, 3]

Interpretation:

    8 learned filters
    each observing 3 RGB channels
"""
# %%


# ============================================================
# FILTER WEIGHTS
# ============================================================

filter_weights = convolution_layer.weight

print("\n[FILTER WEIGHT INFORMATION]")
print("-" * 50)

print(f"Weight Shape : {filter_weights.shape}")


# %% [markdown]
"""
# PARAMETER COUNT

Neural systems learn by adjusting parameters.

The filters themselves are learnable tensor structures.
"""
# %%


# ============================================================
# PARAMETER COUNT
# ============================================================

total_parameters = sum(
    parameter.numel()
    for parameter in convolution_layer.parameters()
)

print("\n[PARAMETER INFORMATION]")
print("-" * 50)

print(f"Total Parameters : {total_parameters}")


# %% [markdown]
"""
# SPATIAL LEARNING INTUITION

The network does not memorize entire images.

It gradually develops:

- reusable local detectors
- hierarchical spatial patterns
- distributed representations

Early CNN layers usually learn:

- edges
- gradients
- textures

Later layers learn:

- shapes
- object parts
- semantic structures
"""
# %%


# ============================================================
# VISUALIZE ORIGINAL IMAGE
# ============================================================

original_image = image_tensor[0]

original_image = original_image.permute(1, 2, 0)

plt.figure(figsize=(6, 6))

plt.imshow(original_image)

plt.title("Original Image")

plt.axis("off")

plt.show()


# %% [markdown]
"""
# MEMORY INFORMATION

Feature extraction itself is tensor transformation.

The network continuously transforms:

    tensor geometry
        ↓
    feature geometry
        ↓
    representation geometry
"""
# %%


# ============================================================
# MEMORY INFORMATION
# ============================================================

num_elements = activated_features.numel()

memory_bytes = (
    num_elements *
    activated_features.element_size()
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

Convolution layers transform:

    raw image tensors
        ↓
    spatial feature systems

This is one of the foundational mechanisms behind:

- computer vision
- CNNs
- representation learning
- modern deep learning systems
"""
# %%