
# %% [markdown]
"""
# PLAY WITH KALU
# STEP 07 — MODEL EVALUATION

## CORE INTUITION

Training measures learning.

Evaluation measures understanding.

A neural network may memorize training samples
without learning generalized representations.

Evaluation helps us measure:

- prediction quality
- generalization ability
- classification performance
- representation reliability

This step introduces:

- evaluation mode
- accuracy
- prediction analysis
- confusion intuition
- inference systems
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
# CONFIGURATION
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

The same preprocessing pipeline used during training
must also be used during evaluation.
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
# DATASET AND DATALOADER
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

    shuffle=False
)

print("\n[DATASET INFORMATION]")
print("-" * 50)

print(f"Dataset Size : {len(dataset)}")

print(f"Classes      : {dataset.classes}")


# %% [markdown]
"""
# SIMPLE CNN CLASSIFIER

For simplicity we recreate the same architecture from
the previous step.
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

            nn.MaxPool2d(2),

            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(2)
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
# MODEL INITIALIZATION
"""
# %%


# ============================================================
# INITIALIZE MODEL
# ============================================================

num_classes = len(dataset.classes)

model = SimpleCNN(
    num_classes=num_classes
)

print("\n[MODEL INFORMATION]")
print("-" * 50)

print(model)


# %% [markdown]
"""
# EVALUATION MODE

`model.eval()` changes model behavior.

Important for layers such as:

- dropout
- batch normalization

Evaluation mode ensures deterministic inference behavior.
"""
# %%


# ============================================================
# EVALUATION MODE
# ============================================================

model.eval()

print("\n[MODEL MODE]")
print("-" * 50)

print("Model set to evaluation mode.")


# %% [markdown]
"""
# DISABLING GRADIENTS

During evaluation we do not compute gradients.

Benefits:

- reduced memory usage
- faster inference
- computational efficiency
"""
# %%


# ============================================================
# EVALUATION LOOP
# ============================================================

correct_predictions = 0

total_samples = 0

all_predictions = []

all_labels = []

with torch.no_grad():

    for batch_images, batch_labels in dataloader:

        logits = model(batch_images)

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        correct_predictions += (
            predictions == batch_labels
        ).sum().item()

        total_samples += batch_labels.size(0)

        all_predictions.extend(
            predictions.tolist()
        )

        all_labels.extend(
            batch_labels.tolist()
        )


# %% [markdown]
"""
# ACCURACY

Accuracy measures:

    correct predictions
            ÷
    total predictions
"""
# %%


# ============================================================
# ACCURACY CALCULATION
# ============================================================

accuracy = (
    correct_predictions /
    total_samples
)

print("\n[EVALUATION RESULTS]")
print("-" * 50)

print(f"Total Samples       : {total_samples}")

print(
    f"Correct Predictions : "
    f"{correct_predictions}"
)

print(f"Accuracy             : {accuracy:.4f}")


# %% [markdown]
"""
# CLASS-WISE ANALYSIS

We analyze predictions for each semantic category.
"""
# %%


# ============================================================
# CLASS-WISE ACCURACY
# ============================================================

class_correct = {
    class_name: 0
    for class_name in dataset.classes
}

class_total = {
    class_name: 0
    for class_name in dataset.classes
}

for prediction, label in zip(
    all_predictions,
    all_labels
):

    class_name = dataset.classes[label]

    class_total[class_name] += 1

    if prediction == label:

        class_correct[class_name] += 1


print("\n[CLASS-WISE ACCURACY]")
print("-" * 50)

for class_name in dataset.classes:

    total = class_total[class_name]

    correct = class_correct[class_name]

    if total > 0:

        class_accuracy = correct / total

    else:

        class_accuracy = 0.0

    print(
        f"{class_name:<10} : "
        f"{class_accuracy:.4f}"
    )


# %% [markdown]
"""
# PREDICTION INTUITION

Predictions emerge from representation similarity.

The classifier does not explicitly understand:

- dogs
- cats
- birds

Instead it learns statistical feature structures
associated with each category.
"""
# %%


# ============================================================
# FETCH ONE BATCH FOR VISUALIZATION
# ============================================================

batch_images, batch_labels = next(
    iter(dataloader)
)

with torch.no_grad():

    logits = model(batch_images)

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    predictions = torch.argmax(
        probabilities,
        dim=1
    )


# %% [markdown]
"""
# REVERSE NORMALIZATION

The images were normalized earlier.

We reverse normalization for visualization.
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
# VISUALIZE PREDICTIONS

Visualization helps connect:

- tensor predictions
- semantic interpretation
- model behavior
"""
# %%


# ============================================================
# VISUALIZATION
# ============================================================

figure, axis = plt.subplots(
    nrows=1,
    ncols=BATCH_SIZE,
    figsize=(15, 5)
)

for index in range(BATCH_SIZE):

    image = visual_batch[index]

    image = image.permute(1, 2, 0)

    predicted_index = predictions[index]

    actual_index = batch_labels[index]

    predicted_name = dataset.classes[
        predicted_index
    ]

    actual_name = dataset.classes[
        actual_index
    ]

    axis[index].imshow(image)

    axis[index].set_title(
        f"P: {predicted_name}\n"
        f"A: {actual_name}"
    )

    axis[index].axis("off")

plt.tight_layout()

plt.show()


# %% [markdown]
"""
# CONFUSION INTUITION

Misclassifications reveal:

- representation overlap
- feature ambiguity
- insufficient learning
- dataset limitations

Example:

A dog may resemble another breed
through shared visual features.
"""
# %%


# ============================================================
# PARAMETER INFORMATION
# ============================================================

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print("\n[PARAMETER INFORMATION]")
print("-" * 50)

print(f"Total Parameters : {total_parameters}")


# %% [markdown]
"""
# MEMORY INFORMATION

Inference is also tensor transformation.

The network continuously transforms:

    image tensors
        ↓
    feature tensors
        ↓
    logits
        ↓
    probabilities
"""
# %%


# ============================================================
# MEMORY INFORMATION
# ============================================================

num_elements = probabilities.numel()

memory_bytes = (
    num_elements *
    probabilities.element_size()
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

Evaluation measures how well the network generalizes.

The model transforms:

    image tensors
        ↓
    learned representations
        ↓
    probabilistic predictions

Evaluation helps determine whether the system has learned:

- reusable visual structure
rather than
- memorized examples

This is one of the foundational ideas behind reliable AI systems.
"""
# %%