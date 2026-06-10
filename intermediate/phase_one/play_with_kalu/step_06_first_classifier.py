
# %% [markdown]
"""
# PLAY WITH KALU
# STEP 06 — FIRST CLASSIFIER

## CORE INTUITION

A classifier attempts to map:

    tensor representations
        ↓
    semantic categories

Example:

    image tensor
        ↓
    dog / cat / bird

The network gradually learns:

- visual representations
- feature hierarchies
- category separation

This step introduces:

- convolutional neural networks
- logits
- softmax probabilities
- classification
- loss functions
- prediction systems
"""
# %%


# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path

import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from torchvision import datasets
from torchvision import transforms

from torch.utils.data import DataLoader


# %% [markdown]
"""
# DATASET CONFIGURATION
"""
# %%


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_PATH = Path("dataset")

IMAGE_SIZE = 224

BATCH_SIZE = 4


# %% [markdown]
"""
# PREPROCESSING PIPELINE

Images are transformed into stable tensor structures.
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

Each folder becomes a semantic category.

Example:

    dogs/
    cats/
    birds/
"""
# %%


# ============================================================
# CREATE DATASET
# ============================================================

dataset = datasets.ImageFolder(
    root=DATASET_PATH,
    transform=transform_pipeline
)

dataloader = DataLoader(
    dataset=dataset,

    batch_size=BATCH_SIZE,

    shuffle=True
)

print("\n[DATASET INFORMATION]")
print("-" * 50)

print(f"Dataset Size : {len(dataset)}")

print(f"Classes      : {dataset.classes}")

print(f"Number of Classes : {len(dataset.classes)}")


# %% [markdown]
"""
# FETCH ONE BATCH

A mini-batch contains:

- image tensor population
- label tensor population
"""
# %%


# ============================================================
# FETCH BATCH
# ============================================================

batch_images, batch_labels = next(
    iter(dataloader)
)

print("\n[BATCH INFORMATION]")
print("-" * 50)

print(f"Images Shape : {batch_images.shape}")

print(f"Labels Shape : {batch_labels.shape}")


# %% [markdown]
"""
# SIMPLE CNN CLASSIFIER

The classifier contains:

1. Convolution layers
2. Activation functions
3. Pooling operations
4. Fully connected layers

Pipeline:

    image tensor
        ↓
    feature extraction
        ↓
    compressed representation
        ↓
    category prediction
"""
# %%


# ============================================================
# SIMPLE CNN CLASSIFIER
# ============================================================

class SimpleCNN(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        self.feature_extractor = nn.Sequential(

            nn.Conv2d(
                in_channels=3,
                out_channels=8,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2
            ),

            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2
            )
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                16 * 56 * 56,
                128
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                num_classes
            )
        )

    def forward(self, x):

        x = self.feature_extractor(x)

        x = self.classifier(x)

        return x


# %% [markdown]
"""
# INITIALIZE MODEL
"""
# %%


# ============================================================
# MODEL INITIALIZATION
# ============================================================

num_classes = len(dataset.classes)

model = SimpleCNN(
    num_classes=num_classes
)

print("\n[MODEL ARCHITECTURE]")
print("-" * 50)

print(model)


# %% [markdown]
"""
# FORWARD PASS

The model transforms:

    image tensors
        ↓
    logits

Logits are raw prediction scores.
"""
# %%


# ============================================================
# FORWARD PASS
# ============================================================

logits = model(batch_images)

print("\n[MODEL OUTPUT]")
print("-" * 50)

print(f"Logits Shape : {logits.shape}")


# %% [markdown]
"""
# LOGIT GEOMETRY

Example:

    [4, 3]

means:

    4 images
    3 class prediction scores per image
"""
# %%


# ============================================================
# LOGIT GEOMETRY
# ============================================================

batch_size, num_outputs = logits.shape

print("\n[LOGIT GEOMETRY]")
print("-" * 50)

print(f"Batch Size : {batch_size}")

print(f"Class Scores Per Image : {num_outputs}")


# %% [markdown]
"""
# SOFTMAX PROBABILITIES

Softmax converts logits into probabilities.

Formula:

                 e^x
    probability = ------
                  Σ(e^x)

Output probabilities sum to 1.
"""
# %%


# ============================================================
# SOFTMAX PROBABILITIES
# ============================================================

probabilities = torch.softmax(
    logits,
    dim=1
)

print("\n[SOFTMAX PROBABILITIES]")
print("-" * 50)

print(probabilities)


# %% [markdown]
"""
# PREDICTIONS

The predicted class corresponds to the maximum probability.
"""
# %%


# ============================================================
# PREDICTIONS
# ============================================================

predicted_classes = torch.argmax(
    probabilities,
    dim=1
)

print("\n[PREDICTIONS]")
print("-" * 50)

for index in range(batch_size):

    predicted_index = predicted_classes[index]

    predicted_name = dataset.classes[predicted_index]

    actual_index = batch_labels[index]

    actual_name = dataset.classes[actual_index]

    print(
        f"Image {index + 1}"
    )

    print(
        f"Predicted : {predicted_name}"
    )

    print(
        f"Actual    : {actual_name}"
    )

    print("-" * 30)


# %% [markdown]
"""
# LOSS FUNCTION

The loss function measures prediction error.

Cross Entropy Loss is commonly used for classification.
"""
# %%


# ============================================================
# LOSS FUNCTION
# ============================================================

loss_function = nn.CrossEntropyLoss()

loss = loss_function(
    logits,
    batch_labels
)

print("\n[LOSS INFORMATION]")
print("-" * 50)

print(f"Loss Value : {loss.item():.4f}")


# %% [markdown]
"""
# WHY LOSS MATTERS

The network learns by minimizing loss.

Training Loop:

    prediction
        ↓
    error measurement
        ↓
    gradient computation
        ↓
    parameter update

This process gradually improves representation quality.
"""
# %%


# ============================================================
# VISUALIZE PREDICTIONS
# ============================================================

visual_batch = batch_images.clone()

visual_batch = (
    visual_batch * 0.5
) + 0.5

figure, axis = plt.subplots(
    nrows=1,
    ncols=BATCH_SIZE,
    figsize=(15, 5)
)

for index in range(BATCH_SIZE):

    image = visual_batch[index]

    image = image.permute(1, 2, 0)

    predicted_index = predicted_classes[index]

    predicted_name = dataset.classes[predicted_index]

    axis[index].imshow(image)

    axis[index].set_title(
        f"Prediction:\n{predicted_name}"
    )

    axis[index].axis("off")

plt.tight_layout()

plt.show()


# %% [markdown]
"""
# PARAMETER INFORMATION

Neural systems learn by adjusting tensor parameters.
"""
# %%


# ============================================================
# PARAMETER INFORMATION
# ============================================================

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

trainable_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)

print("\n[PARAMETER INFORMATION]")
print("-" * 50)

print(f"Total Parameters     : {total_parameters}")

print(f"Trainable Parameters : {trainable_parameters}")


# %% [markdown]
"""
# MEMORY INFORMATION
"""
# %%


# ============================================================
# MEMORY INFORMATION
# ============================================================

num_elements = logits.numel()

memory_bytes = (
    num_elements *
    logits.element_size()
)

print("\n[MEMORY INFORMATION]")
print("-" * 50)

print(f"Tensor Elements : {num_elements}")

print(
    f"Approx Memory   : "
    f"{memory_bytes:.2f} Bytes"
)


# %% [markdown]
"""
# FINAL CONCEPTUAL SUMMARY

A classifier transforms:

    image tensors
        ↓
    feature representations
        ↓
    semantic predictions

The network gradually learns category separation through:

- feature extraction
- representation compression
- error correction
- parameter optimization

This is one of the foundational ideas behind modern AI systems.
"""
# %%