# PLAY WITH KALU

## Overview

`play_with_kalu` is a small educational computer vision experiment inside the `pytorch-self-learning` repository.

The project explores how deep learning systems gradually learn visual identity through tensor transformations.

Rather than treating neural networks as black-box magic, this project focuses on understanding:

* images as tensors
* channels as information fields
* preprocessing pipelines
* batch systems
* convolutional feature learning
* classification systems
* evaluation and generalization

The primary educational goal is:

```text
Tensor intuition first.
Framework syntax second.
```

---

# Educational Philosophy

Most beginner tutorials directly jump into:

```text
Image
    ↓
CNN
    ↓
Prediction
```

This project instead explores the intermediate computational stages:

```text
Image
    ↓
Tensor Geometry
    ↓
Feature Extraction
    ↓
Representation Learning
    ↓
Classification
```

The goal is not merely to build a classifier.

The goal is to understand:

```text
How machines gradually learn visual structure.
```

---

# Learning Outcomes

After completing this mini project, learners should understand:

## Tensor Foundations

* image tensors
* tensor geometry
* channel systems
* spatial dimensions
* tensor memory structures

---

## Preprocessing Systems

* resizing
* normalization
* augmentation
* preprocessing pipelines

---

## Batch Systems

* mini-batch learning
* batch dimensions
* tensor populations
* dataloaders

---

## Feature Learning

* convolution intuition
* local feature extraction
* activation maps
* spatial pattern learning

---

## Classification Systems

* logits
* softmax probabilities
* CNN architectures
* prediction pipelines
* semantic categories

---

## Evaluation Systems

* model evaluation
* accuracy
* generalization
* inference
* prediction analysis

---

# Project Structure

```text
play_with_kalu/

├── dataset/
│   ├── dogs/
│   ├── cats/
│   ├── birds/
│   └── others/
│
├── step_01_image_as_tensor.py
├── step_02_understanding_channels.py
├── step_03_tensor_preprocessing.py
├── step_04_batching_images.py
├── step_05_simple_feature_learning.py
├── step_06_first_classifier.py
├── step_08_model_evaluation.py
│
└── readme.md
```

---

# Dataset Structure

Images are grouped by semantic category.

Example:

```text
dataset/

├── dogs/
├── cats/
├── birds/
└── others/
```

Each folder name automatically becomes a class label.

---

# Installation

## Create Virtual Environment

```bash
python -m venv venv
```

---

## Activate Environment

### Linux / macOS

```bash
source venv/bin/activate
```

### Windows

```bash
venv\Scripts\activate
```

---

## Install Dependencies

### CPU Installation

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Install Additional Packages

```bash
pip install -r requirements.txt
```

---

# How to Run

Run each script sequentially.

Example:

```bash
python step_01_image_as_tensor.py
```

Then continue:

```bash
python step_02_understanding_channels.py
```

and so on.

---

# Learning Path

## Step 01 — Image as Tensor

Core idea:

```text
An image becomes structured numerical geometry.
```

Topics:

* image tensors
* tensor shapes
* channels
* numerical representations

---

## Step 02 — Understanding Channels

Core idea:

```text
Channels are separate information fields.
```

Topics:

* RGB decomposition
* channel extraction
* tensor planes
* color information

---

## Step 03 — Tensor Preprocessing

Core idea:

```text
Learning systems require geometric consistency.
```

Topics:

* resizing
* normalization
* preprocessing pipelines
* augmentation intuition

---

## Step 04 — Batching Images

Core idea:

```text
Neural systems learn from tensor populations.
```

Topics:

* dataloaders
* mini-batches
* batch dimensions
* tensor populations

---

## Step 05 — Simple Feature Learning

Core idea:

```text
Convolution learns local spatial patterns.
```

Topics:

* convolution layers
* feature maps
* activations
* spatial learning

---

## Step 06 — First Classifier

Core idea:

```text
Classification emerges from learned representations.
```

Topics:

* CNNs
* logits
* softmax
* semantic prediction

---

## Step 08 — Model Evaluation

Core idea:

```text
Evaluation measures generalization.
```

Topics:

* inference
* accuracy
* evaluation mode
* prediction analysis

---

# Important Conceptual Insight

A neural network does not initially understand:

* dogs
* cats
* birds

It first learns:

* edges
* gradients
* textures
* statistical spatial structures

Modern deep learning systems emerge from layered tensor transformations.

---

# Intended Audience

This project is designed for learners interested in:

* PyTorch
* deep learning
* tensor systems
* scientific computing
* representation learning
* mathematical intuition behind AI

---

# Project Philosophy

This repository views PyTorch as:

```text
a multidimensional computational tensor system
```

rather than merely:

```text
a deep learning framework
```

Strong emphasis is placed on:

* tensor geometry
* computational thinking
* representation structures
* latent learning systems

---

# Future Extensions

Potential future additions:

* breed classification
* embeddings visualization
* latent space projections
* Grad-CAM
* attention systems
* transfer learning
* similarity search
* explainable AI

---

# Final Thought

The purpose of this project is not simply:

```text
"train a classifier"
```

The purpose is to explore:

```text
How tensor systems gradually evolve into machine perception.
```
## Intuition Building Questions

Try to answer the following questions before moving to the next stage.

The goal is not memorization.

The goal is to develop tensor intuition.

---

### Image Tensor Intuition

1. Why does PyTorch store images as:

```text id="zhj88k"
[Channels, Height, Width]
```

instead of:

```text id="3m83na"
[Height, Width, Channels]
```

2. If an image tensor has shape:

```python id="e4vbch"
torch.Size([3, 224, 224])
```

what does each dimension represent?

3. Why are RGB channels separated into different tensor planes?

4. If a grayscale image is used instead of RGB:

* how would tensor shape change?
* how would memory usage change?

---

### Tensor Geometry Intuition

5. Why does resizing become necessary before batching?

6. What problems occur if different image sizes exist inside the same batch?

7. Why does adding a batch dimension transform:

```text id="rrt92t"
[3, 224, 224]
```

into:

```text id="qhhv8g"
[1, 3, 224, 224]
```

8. Why is batching computationally more efficient than processing one image at a time?

---

### Convolution Intuition

9. What does a convolution filter actually observe?

10. Why do early CNN layers usually learn:

* edges
* gradients
* textures

instead of semantic concepts like:

* dog
* cat
* face

11. If a convolution kernel becomes larger:

* what changes spatially?
* what changes computationally?

12. Why can feature maps be interpreted as learned spatial responses?

---

### Representation Learning Intuition

13. Why are logits not probabilities?

14. Why do we apply softmax after logits?

15. What does the classifier actually learn internally?

Does it learn:

```text id="fpr7lj"
"dogness"
```

or:

```text id="4o0v3u"
statistical feature structure
```

16. Why can two different dog breeds still produce similar representations?

---

### Evaluation Intuition

17. Why is high training accuracy alone not sufficient?

18. What is the difference between:

```text id="f3h7i4"
memorization
```

and:

```text id="hpbpm6"
generalization
```

19. Why is evaluation performed using:

```python id="jmm0di"
torch.no_grad()
```

20. What kinds of visual patterns might confuse the classifier?

---

## Final Reflection

Try to think about the following deeply:

```text id="h5w6qa"
At what stage does numerical tensor processing
begin to resemble perception?
```

This question sits near the center of modern deep learning research.
