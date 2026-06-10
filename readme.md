# PyTorch Self Learning

This structured self-learning journey into PyTorch focused on:

* tensor intuition
* mathematical understanding
* computational thinking
* multidimensional data manipulation
* deep learning foundations

This repository is not intended to be:

* an API dump
* copy-paste tutorial collection
* framework memorization guide

Instead, the goal is to develop:

> operational intuition for tensor systems and computational mathematics.

---

# Philosophy

Most PyTorch tutorials teach:

```text
Framework Syntax → Neural Networks → Hope For Intuition
```

This repository follows a different path:

```text
Tensor Intuition
    ↓
Mathematical Operations
    ↓
Computational Structures
    ↓
Deep Learning Foundations
```

The focus is on:

* understanding tensor behavior
* dimensional reasoning
* reduction systems
* computational graphs
* mathematical structure

before jumping into large models.

---

# Repository Goals

This project aims to build intuition for:

* tensors
* tensor geometry
* tensor manipulation
* automatic differentiation
* multidimensional computation
* linear algebra operations
* reduction systems
* computational graph reasoning

using progressively structured examples.

---

# Topics Covered So Far

# 1. Hello PyTorch

Introduction to:

* `torch`
* tensors
* tensor creation
* tensor properties

Concepts:

* tensor shapes
* datatypes
* devices
* multidimensional arrays

---

# 2. Tensor Creation

Covered APIs:

```python
torch.tensor()
torch.Tensor()
torch.zeros()
torch.ones()
torch.empty()
torch.rand()
torch.randn()
torch.randint()
torch.arange()
torch.linspace()
torch.logspace()
torch.eye()
torch.full()
torch.zeros_like()
torch.ones_like()
torch.rand_like()
torch.randn_like()
```

Concepts:

* memory allocation
* random initialization
* tensor factories
* Gaussian distributions
* identity matrices

---

# 3. Tensor Information

Covered APIs:

```python
tensor.shape
tensor.size()
tensor.dtype
tensor.device
tensor.ndim
tensor.numel()
tensor.requires_grad
```

Concepts:

* tensor metadata
* shape analysis
* dimensionality
* gradient tracking
* device management

---

# 4. Tensor Type Conversion

Covered APIs:

```python
tensor.float()
tensor.double()
tensor.int()
tensor.long()
tensor.bool()
tensor.to()
tensor.cpu()
tensor.cuda()
```

Concepts:

* datatype systems
* precision
* CPU/GPU movement
* memory representation

---

# 5. Tensor Manipulation

Covered APIs:

```python
torch.cat()
torch.stack()
torch.split()
torch.chunk()

tensor.view()
tensor.reshape()
tensor.permute()
tensor.transpose()
tensor.squeeze()
tensor.unsqueeze()
tensor.flatten()
tensor.repeat()
tensor.expand()
```

Concepts:

* shape transformations
* batching
* dimension reordering
* tensor replication
* broadcasting intuition

---

# 6. Mathematical Operations

Covered APIs:

```python
torch.add()
torch.sub()
torch.mul()
torch.div()
torch.matmul()
torch.mm()
torch.bmm()

torch.exp()
torch.log()
torch.sqrt()
torch.pow()
torch.abs()
torch.sin()
torch.cos()
torch.tanh()
torch.relu()
torch.sigmoid()
```

Concepts:

* element-wise operations
* matrix multiplication
* activation functions
* exponential systems
* tensor algebra

---

# 7. Reduction Operations

Covered APIs:

```python
torch.sum()
torch.mean()
torch.std()
torch.var()
torch.max()
torch.min()
torch.argmax()
torch.argmin()
torch.prod()
```

Concepts:

* aggregation
* statistical summaries
* dimensional collapse
* reduction reasoning

---

# 8. Comparison Operations

Covered APIs:

```python
torch.eq()
torch.ne()
torch.gt()
torch.lt()
torch.ge()
torch.le()
torch.where()
```

Concepts:

* masking
* conditional tensor selection
* logical tensor systems
* thresholding

---

# 9. Autograd

Covered APIs:

```python
tensor.backward()
torch.autograd.grad()
torch.no_grad()
torch.enable_grad()
torch.set_grad_enabled()
```

Concepts:

* computational graphs
* automatic differentiation
* chain rule
* gradient propagation
* backpropagation

---

# 10. Variable Operations

Covered APIs:

```python
tensor.requires_grad_()
tensor.detach()
tensor.clone()
```

Concepts:

* graph detachment
* gradient control
* memory copying
* tensor tracking

---

# 11. Beginner Tensor Problems

Implemented beginner-friendly problems involving:

* tensor analysis
* reductions
* squeeze/unsqueeze
* averaging
* batching
* indexing

Example:

* Student Marks Tensor Analysis

---

# 12. Intermediate Tensor Systems Problem

Implemented:

* separable Gaussian filtering
* multidimensional tensor filtering
* grouped convolutions
* channel-wise operations

Concepts:

* tensor geometry
* signal processing intuition
* separable convolution
* computational optimization

---

# Educational Design

The repository emphasizes:

```text
Concept
    ↓
Mathematics
    ↓
Tensor Operation
    ↓
Algorithmic Thinking
    ↓
PyTorch Implementation
```

Each section typically includes:

* explanations
* mathematical formulas
* algorithmic intuition
* notebook exercises
* standalone scripts

---

# Why This Repository Is Different

This project intentionally bridges:

| Mathematics       | PyTorch           |
| ----------------- | ----------------- |
| Linear Algebra    | Tensor Operations |
| Calculus          | Autograd          |
| Geometry          | Tensor Shapes     |
| Signal Processing | Convolutions      |
| Optimization      | Gradient Systems  |

instead of treating PyTorch as merely a software library.

---

# Repository Structure

```text
pytorch-self-learning/
│
├── basics/
│   ├── tensor_creation/
│   ├── tensor_information/
│   ├── tensor_manipulation/
│   ├── tensor_math/
│   ├── reductions/
│   ├── comparison_ops/
│   ├── autograd/
│   └── variables/
│
├── problems/
│   ├── beginner/
│   └── intermediate/
│
├── notebooks/
│
└── README.md
```

---
# Intermediate Phase — Tensor Interaction Systems

## Overview

After learning how tensors represent images and perception systems in:

```text id="g3l9dn"
play_with_kalu
```

the repository now transitions into a deeper layer of tensor understanding:

```text id="0n8o3d"
Tensor Interaction Systems
```

This phase focuses on how tensors:

* align
* negotiate dimensions
* route information
* reorganize geometry
* control visibility
* interact structurally

The goal is to move beyond:

```text id="e73z1f"
Tensor as container
```

toward:

```text id="7yk4yl"
Tensor as computational geometry system
```

---

# Core Philosophy

This phase repeatedly emphasizes:

```text id="8l3zci"
Tensor dimensions are not merely sizes.

They carry structural meaning.
```

Understanding tensor interaction systems is foundational for:

* CNNs
* transformers
* attention systems
* latent representations
* multimodal systems
* scientific computing

---

# Directory Structure

```text id="2afq4j"
intermediate/

└── phase_02_tensor_interaction_systems/

    ├── step_01_tensor_broadcasting.py
    ├── step_02_broadcasting_geometry.py
    ├── step_03_tensor_indexing.py
    ├── step_04_boolean_masking.py
    ├── step_05_gather_scatter.py
    ├── step_06_tensor_memory_layout.py
    ├── step_07_tensor_permutation_systems.py
    └── step_08_tensor_alignment_systems.py
```

---

# Learning Progression

This phase follows the progression:

```text id="8g1ud0"
Broadcasting
    ↓
Tensor Navigation
    ↓
Information Visibility
    ↓
Tensor Routing
    ↓
Memory Geometry
    ↓
Dimension Reorganization
    ↓
Representation Alignment
```

---

# Step-by-Step Learning Path

---

## Step 01 — Tensor Broadcasting

Core intuition:

```text id="jjlwm8"
Dimensions negotiate compatibility.
```

Topics:

* singleton dimensions
* implicit expansion
* broadcasting rules
* shape compatibility
* tensor interaction

Key insight:

```text id="ddlf6w"
Broadcasting is structured geometric alignment.
```

---

## Step 02 — Broadcasting Geometry

Core intuition:

```text id="l6e6o4"
Broadcasting is tensor geometry negotiation.
```

Topics:

* right-to-left alignment
* dimensional compatibility
* higher-dimensional broadcasting
* geometric expansion
* structural interaction

Key insight:

```text id="mjdb55"
Tensor interaction depends on geometric compatibility.
```

---

## Step 03 — Tensor Indexing

Core intuition:

```text id="uh54te"
Indexing is tensor navigation.
```

Topics:

* tensor slicing
* dimensional traversal
* advanced indexing
* boolean indexing
* higher-dimensional indexing

Key insight:

```text id="y4br9n"
Tensor indexing enables structured information access.
```

---

## Step 04 — Boolean Masking

Core intuition:

```text id="a0zjlwm"
Masks control information flow.
```

Topics:

* boolean tensors
* logical masking
* conditional selection
* masked computation
* sparse visibility

Key insight:

```text id="5rjlwm"
Masking enables selective tensor visibility.
```

---

## Step 05 — Gather and Scatter

Core intuition:

```text id="zjlwm8"
Tensors can dynamically route information.
```

Topics:

* gather()
* scatter()
* tensor routing
* sparse interaction
* dynamic indexing

Key insight:

```text id="jqjlwm"
Tensor systems can selectively redistribute information.
```

---

## Step 06 — Tensor Memory Layout

Core intuition:

```text id="jlwmu5"
Tensor geometry affects memory behavior.
```

Topics:

* contiguous tensors
* tensor storage
* memory layout
* tensor stride
* reshape vs view

Key insight:

```text id="jlwmu6"
A tensor is simultaneously geometry and memory.
```

---

## Step 07 — Tensor Permutation Systems

Core intuition:

```text id="jlwmu7"
Dimension order changes tensor meaning.
```

Topics:

* permute()
* transpose()
* dimension reordering
* semantic structure
* tensor reinterpretation

Key insight:

```text id="jlwmu8"
Tensor dimensions encode semantic structure.
```

---

## Step 08 — Tensor Alignment Systems

Core intuition:

```text id="jlwmu9"
Deep learning is structured tensor interaction.
```

Topics:

* tensor alignment
* representation compatibility
* interaction geometry
* embedding alignment
* attention alignment

Key insight:

```text id="jlwmua"
Tensor systems interact through structural negotiation.
```

---

# Important Conceptual Shift

Phase 1 taught:

```text id="jlwmub"
How tensors represent perception.
```

Phase 2 teaches:

```text id="jlwmuc"
How tensors structurally interact.
```

This transition is extremely important because modern AI systems fundamentally depend on:

* tensor alignment
* tensor routing
* dimension organization
* latent interaction systems

---

# Why This Phase Matters

Most tutorials stop at:

* tensor creation
* tensor arithmetic
* basic neural networks

This repository intentionally explores:

```text id="jlwmud"
The computational geometry of tensor systems.
```

These ideas form the foundation for understanding:

* transformers
* attention
* embeddings
* multimodal systems
* latent spaces
* representation learning

---

# Important Deep Learning Connection

Nearly every modern AI architecture depends heavily on:

* broadcasting
* masking
* permutation
* alignment
* tensor routing

Understanding these deeply transforms how one understands AI systems.

---

# Final Conceptual Summary

This phase gradually reveals that deep learning systems are fundamentally:

```text id="jlwmue"
large-scale tensor interaction systems
```

where dimensions carry:

* structure
* semantics
* geometry
* interaction meaning

rather than merely:

```text id="jlwmuf"
numerical storage.
```


# Future Topics

Planned future sections include:

* convolution operations
* neural networks
* CNNs
* transformers
* attention mechanisms
* optimization
* CUDA programming
* custom autograd
* latent representations
* tensor geometry
* diffusion systems

---

# Recommended Audience

This repository is especially useful for:

* self learners
* engineers transitioning from MATLAB
* scientific computing learners
* ML beginners wanting intuition
* researchers building tensor understanding

---

# Final Note

PyTorch is more than a deep learning framework.

It is fundamentally:

> a multidimensional computational mathematics system.

The goal of this repository is to learn PyTorch not merely as:

* syntax

but as:

* tensor reasoning
* computational structure
* mathematical abstraction
* differentiable systems engineering.
 

# How To Use

## 1. Clone Repository

```bash id="h1"
git clone git@github.com:kncsolutions/pytorch-self-learning.git
```

Enter project directory:

```bash id="h2"
cd pytorch-self-learning
```

---

# 2. Create Virtual Environment (Recommended)

## Using `venv`

```bash id="h3"
python -m venv venv
```

Activate environment:

### Linux / macOS

```bash id="h4"
source venv/bin/activate
```

### Windows

```bash id="h5"
venv\Scripts\activate
```

---

# 3. Install Dependencies

Install PyTorch.

Visit:

```text id="h6"
https://pytorch.org/get-started/locally/
```

or install CPU version directly:

```bash id="h7"
pip install torch torchvision torchaudio
```

Install notebook support:

```bash id="h8"
pip install notebook matplotlib
```

---

# 4. Run Python Scripts

Example:

```bash id="h9"
python basics/tensor_creation.py
```

Run mathematical operations:

```bash id="h10"
python basics/tensor_mathematics.py
```

Run tensor manipulation examples:

```bash id="h11"
python basics/tensor_manipulation.py
```

---

# 5. Launch Jupyter Notebook

Start notebook server:

```bash id="h12"
jupyter notebook
```

Open notebook examples from:

```text id="h13"
notebooks/
```

Example notebooks:

```text id="h14"
notebooks/basics/tensor_creation.ipynb

notebooks/basics/tensor_mathematics.ipynb

notebooks/basics/tensor_manipulation.ipynb
```

---

# 6. Solve Beginner Problems

Problem files are available in:

```text id="h15"
problems/beginner/
```

Example:

```bash id="h16"
python problems/beginner/tensor_basic_task_easy.py
```

---

# 7. Check Solutions

Solutions are available in:

```text id="h17"
solutions/
```

Example:

```bash id="h18"
python solutions/tensor_basic_task_easy_solution.py
```

---

# 8. Intermediate Demonstrations

Intermediate tensor systems demonstrations are available in:

```text 
problems/intermediate/
```

Example:

```
python problems/intermediate/tensor_basic_demonstration_advanced.py
```

---

# Recommended Learning Flow

Recommended progression:

```text id="h21"
1. Read notebook explanation
        ↓
2. Run standalone script
        ↓
3. Modify tensor values
        ↓
4. Observe tensor behavior
        ↓
5. Solve beginner problem
        ↓
6. Compare with solution
        ↓
7. Experiment independently
```

---

# Suggested Execution Order

## Basics

```text id="h22"
1. hello_pytorch.py
2. tensor_creation.py
3. tensor_information.py
4. tensor_type_conversion.py
5. tensor_manipulation.py
6. tensor_mathematics.py
7. tensor_reduction.py
8. tensor_comparison.py
9. tensor_autograd_playground.py
10. tensor_variable_operations.py
```

---

# Important Recommendation

Do NOT merely read the code.

Try:

* changing tensor shapes
* modifying dimensions
* changing datatypes
* experimenting with reductions
* intentionally creating errors

Tensor intuition develops through experimentation.

---

# Development Environment

Current project structure is compatible with:

* Jupyter Notebook
* VSCode
* PyCharm
* Spyder
* Linux terminal workflows

---
````markdown id="wqlycg"
## Installation

### CPU Installation

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
````

### Install Additional Dependencies

```bash
pip install -r requirements.txt
```

```
```

# Verify Installation

Run:

```
python
```

Then:

```
import torch

print(torch.__version__)
```

If PyTorch imports successfully, setup is complete.

# Developed By

Developed by:

```text id="d1"
Pallav Nandi Chaudhuri
```

This repository is part of a structured self-learning and research-oriented exploration into:

* PyTorch
* tensor systems
* computational mathematics
* deep learning foundations
* multidimensional computation
* differentiable systems

---

# Contact / Further Guidance

For:

* questions
* suggestions
* collaborations
* corrections
* learning discussions
* research-oriented conversations

feel free to connect through the repository discussions/issues section.

If this repository helps your learning journey, consider:

* starring the repository
* contributing improvements
* extending exercises
* experimenting independently

---

# Final Note

The objective of this repository is not merely:

* learning APIs

but developing:

* tensor intuition
* computational reasoning
* multidimensional thinking
* mathematical understanding of deep learning systems.

Learning PyTorch deeply means learning:

> how modern AI systems manipulate structured numerical geometry.
> 
> # License

This project is licensed under the MIT License.

See:
- LICENSE
- DISCLAIMER.md
