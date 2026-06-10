
# %% [markdown]
"""
# PLAY WITH KALU
# STEP 04 — BATCHING IMAGES

## CORE INTUITION

Neural networks rarely learn from a single image.

They learn from populations of tensors.

Batching introduces a new tensor dimension:

    [Batch, Channels, Height, Width]

Example:

    [32, 3, 224, 224]

Where:

    32   -> number of images
    3    -> RGB channels
    224  -> height
    224  -> width

This step introduces:

- datasets
- dataloaders
- batch dimensions
- mini-batch learning
- tensor populations
"""
# %%


# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path

import matplotlib.pyplot as plt

import torch

from torchvision import datasets
from torchvision import transforms

from torch.utils.data import DataLoader


# %% [markdown]
"""
# DATASET STRUCTURE

Expected Structure:

dataset/

├── dogs/
├── cats/
├── birds/
└── others/

Each folder name becomes a class label.
"""
# %%


# ============================================================
# DATASET CONFIGURATION
# ============================================================

DATASET_PATH = Path("dataset")

BATCH_SIZE = 4

IMAGE_SIZE = 224


# %% [markdown]
"""
# PREPROCESSING PIPELINE

All images are transformed into geometrically consistent tensors.

Pipeline:

    Resize
        ↓
    Tensor Conversion
        ↓
    Normalization
"""
# %%


# ============================================================
# IMAGE TRANSFORMATIONS
# ============================================================

transform_pipeline = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.5, 0.5, 0.5],
            std=[0.5, 0.5, 0.5]
        )
    ]
)


# %% [markdown]
"""
# IMAGE DATASET

`ImageFolder` automatically:

- reads image folders
- assigns labels
- builds dataset structure

Example:

    dogs  -> 0
    cats  -> 1
"""
# %%


# ============================================================
# CREATE DATASET
# ============================================================

dataset = datasets.ImageFolder(
    root=DATASET_PATH,
    transform=transform_pipeline
)

print("\n[DATASET INFORMATION]")
print("-" * 50)

print(f"Dataset Size : {len(dataset)}")

print(f"Classes      : {dataset.classes}")

print(f"Class Mapping: {dataset.class_to_idx}")


# %% [markdown]
"""
# DATALOADER

The DataLoader creates batches of tensors.

Instead of:

    one image at a time

the network receives:

    multiple tensor samples simultaneously

Benefits:

- computational efficiency
- stable gradient estimation
- parallel learning
"""
# %%


# ============================================================
# CREATE DATALOADER
# ============================================================

dataloader = DataLoader(
    dataset=dataset,

    batch_size=BATCH_SIZE,

    shuffle=True
)

print("\n[DATALOADER INFORMATION]")
print("-" * 50)

print(f"Batch Size : {BATCH_SIZE}")


# %% [markdown]
"""
# EXTRACT FIRST BATCH

A batch contains:

    images tensor
    labels tensor
"""
# %%


# ============================================================
# FETCH ONE BATCH
# ============================================================

batch_images, batch_labels = next(
    iter(dataloader)
)

print("\n[BATCH INFORMATION]")
print("-" * 50)

print(f"Batch Shape  : {batch_images.shape}")

print(f"Labels Shape : {batch_labels.shape}")


# %% [markdown]
"""
# BATCH GEOMETRY

Batch Tensor Shape:

    [Batch, Channels, Height, Width]

Example:

    [4, 3, 224, 224]

Interpretation:

    4 images
    each image contains:
        3 channels
        224 height
        224 width
"""
# %%


# ============================================================
# BATCH GEOMETRY
# ============================================================

batch_size, channels, height, width = (
    batch_images.shape
)

print("\n[BATCH GEOMETRY]")
print("-" * 50)

print(f"Batch Size : {batch_size}")

print(f"Channels   : {channels}")

print(f"Height     : {height}")

print(f"Width      : {width}")


# %% [markdown]
"""
# LABEL INTERPRETATION

Numerical labels correspond to semantic categories.

Example:

    0 -> dogs
    1 -> cats

Neural networks internally operate on numerical representations.
"""
# %%


# ============================================================
# LABEL INFORMATION
# ============================================================

print("\n[LABEL INFORMATION]")
print("-" * 50)

for index, label in enumerate(batch_labels):

    class_name = dataset.classes[label]

    print(
        f"Image {index + 1} "
        f"-> Label: {label.item()} "
        f"({class_name})"
    )


# %% [markdown]
"""
# VISUALIZATION PREPARATION

The images were normalized earlier.

To visualize them correctly we reverse normalization.
"""
# %%


# ============================================================
# REVERSE NORMALIZATION
# ============================================================

visual_batch = batch_images.clone()

visual_batch = (
    visual_batch * 0.5
) + 0.5


# %% [markdown]
"""
# VISUALIZING THE BATCH

This helps visualize:

- tensor populations
- batch organization
- multiple learning samples
"""
# %%


# ============================================================
# VISUALIZE BATCH
# ============================================================

figure, axis = plt.subplots(
    nrows=1,
    ncols=BATCH_SIZE,
    figsize=(15, 5)
)

for index in range(BATCH_SIZE):

    image = visual_batch[index]

    image = image.permute(1, 2, 0)

    label = batch_labels[index]

    class_name = dataset.classes[label]

    axis[index].imshow(image)

    axis[index].set_title(class_name)

    axis[index].axis("off")

plt.tight_layout()

plt.show()


# %% [markdown]
"""
# WHY MINI-BATCH LEARNING EXISTS

Training on entire datasets simultaneously may be:

- memory expensive
- computationally slow

Mini-batches provide a balance between:

- learning stability
- computational efficiency

This is one of the core engineering ideas in deep learning.
"""
# %%


# ============================================================
# MEMORY INFORMATION
# ============================================================

num_elements = batch_images.numel()

memory_bytes = (
    num_elements *
    batch_images.element_size()
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

Deep learning systems do not learn from isolated tensors.

They learn from:

    tensor populations

Batching introduces:

    collective tensor learning

This is the beginning of large-scale representation learning.
"""
# %%