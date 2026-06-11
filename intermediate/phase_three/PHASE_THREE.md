# How To Use — Phase 03 Tensor Fields

## Important Philosophy

This phase is NOT designed to teach:

```text
copy-paste CNN APIs
```

It is designed to teach:

```text
how local tensor interactions gradually
become representation systems
```

You are strongly encouraged to:

* run every file step-by-step
* modify tensors manually
* experiment with kernels
* predict outputs before execution
* observe interaction geometry carefully

Do not rush through the modules.

This phase contains many foundational ideas that
reappear later in:

* CNN systems
* transformers
* diffusion systems
* multimodal systems
* scientific AI systems

---

# Recommended Learning Strategy

For every module follow this sequence:

```text
Read comments carefully
        ↓
Visualize tensor interaction
        ↓
Predict output mentally
        ↓
Run code
        ↓
Observe representation changes
        ↓
Modify tensors and kernels
        ↓
Re-run experiments
```

The goal is NOT memorization.

The goal is:

```text
tensor interaction intuition
```

---

# Important Mindset

Throughout this phase remember:

```text
Convolution is localized structured interaction.
```

A kernel is NOT merely a filter.

A kernel defines:

```text
local interaction behavior
```

Feature maps are NOT merely detected features.

They are:

```text
interaction response fields
```

Pooling is NOT merely tensor shrinking.

It is:

```text
localized information compression
```

This mindset is extremely important.

---

# Suggested Workflow

## Step 1 — Run the Module

Example:

```bash
python step_01_conv1d_intuition.py
```

or inside Jupyter / VSCode Notebook mode:

```python
#%% blocks
```

Run cell-by-cell.

---

# Step 2 — Observe Tensor Shapes

At every stage ask:

* Why did the shape change?
* What interaction happened?
* Which local regions contributed?
* What information was preserved?
* What information was discarded?

Tensor geometry matters.

---

# Step 3 — Modify Kernels

Experiment aggressively.

Try changing:

```python
[1, 0, -1]
```

to:

```python
[-1, 2, -1]
```

or:

```python
[1, 1, 1]
```

Observe:

* feature map changes
* interaction sensitivity changes
* response propagation changes

This is where intuition develops.

---

# Step 4 — Change Stride and Padding

Experiment with:

* stride = 1
* stride = 2
* padding = 0
* padding = 1

Observe:

* spatial sampling density
* information loss
* output geometry
* receptive field behavior

These ideas become critically important later in:

* architecture design
* optimization
* representation engineering

---

# Step 5 — Think Geometrically

Do NOT think:

```text
API → output
```

Think:

```text
local interaction
    ↓
response propagation
    ↓
representation emergence
```

This phase becomes significantly easier when viewed geometrically.

---

# Recommended Questions While Studying

For every module ask:

## Kernels

* What interaction rule does this kernel define?
* Which local structures become amplified?
* Which structures become suppressed?

---

## Feature Maps

* What does this response field represent?
* Which regions activate strongly?
* Why do different kernels produce different maps?

---

## Pooling

* What information survived?
* What information disappeared?
* Was the compression beneficial?

---

## Receptive Fields

* How much context can this neuron observe?
* How does deeper stacking expand interaction scope?

---

# Important Warning

Do NOT memorize:

* formulas blindly
* tensor shapes blindly
* API syntax blindly

Try to understand:

```text
WHY the tensor system behaves this way
```

That understanding compounds heavily later.

---

# Best Way To Learn This Phase

The best approach is:

```text
modify everything
```

Change:

* tensors
* kernels
* strides
* padding
* pooling windows
* activation functions

Then observe how representation geometry changes.

Deep learning intuition emerges through:

```text
interaction experimentation
```

not passive reading.

---

# Final Learning Goal

By the end of this phase you should gradually start seeing CNNs as:

```text
hierarchical tensor interaction systems
```

rather than:

```text
image-processing APIs
```

That conceptual shift is one of the most important transitions in modern AI learning.
