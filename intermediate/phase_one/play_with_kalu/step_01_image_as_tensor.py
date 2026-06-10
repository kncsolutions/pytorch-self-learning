
"""
PLAY WITH KALU
==============
STEP 01 — IMAGE AS TENSOR

CORE INTUITION
--------------
An image inside a deep learning system becomes
structured numerical geometry.

PyTorch represents an image tensor as:

    [Channels, Height, Width]

Example:
    [3, 224, 224]

Where:
    3     -> RGB Channels
    224   -> Height
    224   -> Width
"""

# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path

from PIL import Image

import torch
from torchvision import transforms


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_PATH = Path("dataset/dogs/d1.jpg")


# ============================================================
# LOAD IMAGE
# ============================================================

image = Image.open(IMAGE_PATH)

print("\n[IMAGE INFORMATION]")
print("-" * 50)

print(f"Image Type   : {type(image)}")
print(f"Image Mode   : {image.mode}")
print(f"Image Size   : {image.size}")



# ============================================================
# IMAGE -> TENSOR
# ============================================================

transform = transforms.ToTensor()

image_tensor = transform(image)

print("\n[TENSOR INFORMATION]")
print("-" * 50)

print(f"Tensor Type  : {type(image_tensor)}")
print(f"Tensor Shape : {image_tensor.shape}")
print(f"Tensor Dtype : {image_tensor.dtype}")


# ============================================================
# TENSOR GEOMETRY
# ============================================================

channels, height, width = image_tensor.shape

print("\n[TENSOR GEOMETRY]")
print("-" * 50)

print(f"Channels     : {channels}")
print(f"Height       : {height}")
print(f"Width        : {width}")


# ============================================================
# PIXEL VALUE ANALYSIS
# ============================================================

print("\n[PIXEL VALUE ANALYSIS]")
print("-" * 50)

print(f"Minimum Value : {image_tensor.min():.4f}")
print(f"Maximum Value : {image_tensor.max():.4f}")

"""
ToTensor() converts pixel values from:

    [0, 255]

to:

    [0.0, 1.0]

This scaling improves numerical stability during training.
"""


# ============================================================
# CHANNEL ANALYSIS
# ============================================================

red_channel = image_tensor[0]
green_channel = image_tensor[1]
blue_channel = image_tensor[2]

print("\n[CHANNEL INFORMATION]")
print("-" * 50)

print(f"Red Channel Shape   : {red_channel.shape}")
print(f"Green Channel Shape : {green_channel.shape}")
print(f"Blue Channel Shape  : {blue_channel.shape}")


# ============================================================
# TENSOR STATISTICS
# ============================================================

print("\n[TENSOR STATISTICS]")
print("-" * 50)

print(f"Mean : {image_tensor.mean():.4f}")
print(f"Std  : {image_tensor.std():.4f}")


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
print(f"Memory Usage    : {memory_bytes / 1024:.2f} KB")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n[FINAL SUMMARY]")
print("-" * 50)

print(
    """
Image
    ↓
Numerical Grid
    ↓
Tensor Geometry
    ↓
Computational Representation

This is the foundation of machine perception.
"""
)
image.show()