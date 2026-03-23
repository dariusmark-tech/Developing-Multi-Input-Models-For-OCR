# DigiNsure OCR — Multi-Modal Insurance Document Classifier

> A PyTorch OCR model that classifies scanned insurance IDs as **primary** or **secondary** using both image data and insurance type — built for DigiNsure Inc.'s document digitisation initiative.

---

## Table of Contents

- [Overview](#overview)
- [What I Learned](#what-i-learned)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Project Structure](#project-structure)
- [Setup & Usage](#setup--usage)
- [Key Concepts](#key-concepts)
- [Results](#results)

---

## Overview

DigiNsure Inc. is digitising historical insurance claim documents. This project trains a **multi-modal OCR classifier** that:

- Takes a **64×64 greyscale scan** of an insurance document as input
- Also takes the **insurance type** (home, life, auto, health, or other) as a second input
- Outputs whether the scanned ID is a `primary_id` or `secondary_id`

Multi-modal learning lets the model capture nuances that neither the image nor the type label could provide alone — for example, certain ID patterns may be exclusive to life or auto policies.

---

## What I Learned

This was my first hands-on project combining **computer vision** and **text features** in a single PyTorch model. Here are the main takeaways:

### Multi-Modal Fusion
Using two different data types (image + categorical text) as inputs to one model. Each modality gets its own processing branch, and the features are **concatenated before the classifier** — this is called *late fusion*.

### Convolutional Layers
`Conv2d` slides learnable filters over an image to detect local patterns (edges, curves, character strokes). Key parameters I worked with:

| Parameter | Value | Why |
|---|---|---|
| `in_channels` | 1 | Greyscale image (single channel) |
| `out_channels` | 16 | Learn 16 different feature detectors |
| `kernel_size` | 3 | Look at 3×3 pixel patches |
| `padding` | 1 | Preserve 64×64 spatial size after conv |

### Tensor Shape Tracking
One of the most important debugging habits — tracing shapes through every layer:

```
Input        →  (B, 1, 64, 64)
Conv2d       →  (B, 16, 64, 64)   # padding=1 preserves size
MaxPool2d(2) →  (B, 16, 32, 32)   # halves spatial dims
Flatten      →  (B, 16384)         # 16 × 32 × 32
```

### Late Fusion with `torch.cat`
```python
combined = torch.cat([img_feat, type_feat], dim=1)  # (B, 16400)
```
`dim=1` concatenates along the **feature dimension**, not the batch — B stays the same.

### Safe File Handling
```python
# ✅ Always use context manager
with open('dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

# ❌ Avoid — leaves file handle open
dataset = pickle.load(open('dataset.pkl', 'rb'))
```

---

## Dataset

| Property | Value |
|---|---|
| Total samples | 100 |
| Image size | 64 × 64 pixels, greyscale |
| Insurance types | `home`, `life`, `auto`, `health`, `other` |
| Type encoding | One-hot vector of shape `(5,)` |
| Labels | `primary_id` (0) / `secondary_id` (1) |
| Format | `.pkl` — serialised `ProjectDataset` object |

Each dataset item is a tuple `(img_tensor, label)` where:
- `img_tensor[0]` — image channel, shape `(1, 64, 64)`, normalised to `[0, 1]`
- `img_tensor[1]` — one-hot type vector, shape `(5,)`

---

## Model Architecture

```
┌─────────────────────────────┐     ┌────────────────────┐
│       IMAGE BRANCH          │     │    TYPE BRANCH     │
│                             │     │                    │
│  Input: (B, 1, 64, 64)      │     │  Input: (B, 5)     │
│         ↓                   │     │        ↓           │
│  Conv2d(1→16, k=3, p=1)     │     │  Linear(5→16)      │
│  ReLU                       │     │  ReLU              │
│  MaxPool2d(2)               │     │                    │
│  Flatten → (B, 16384)       │     │  → (B, 16)         │
└──────────────┬──────────────┘     └────────┬───────────┘
               │                             │
               └──────────┬──────────────────┘
                          ↓
              torch.cat([...], dim=1)
                    (B, 16400)
                          ↓
              Linear(16400 → 256)
                    ReLU
                  Dropout(0.3)
              Linear(256 → 2)
                          ↓
              Logits: (B, 2)
         [primary_id | secondary_id]
```

### Why this design?

- **Separate branches** — image and type are very different data types; processing them separately first gives each branch freedom to learn modality-specific features
- **Late fusion** — concatenating after processing (not at the input) is standard for heterogeneous modalities
- **Dropout(0.3)** — with only 100 training samples, regularisation is critical to prevent overfitting

---

## Project Structure

```
📁 digiNsure-ocr/
├── 📓 notebook.ipynb              # Main Jupyter notebook
├── 🐍 project_utils.py            # ProjectDataset class
├── 📦 ocr_insurance_dataset.pkl   # Serialised dataset
└── 🖼️  digitizing_team.png        # Team asset
```

---

## Setup & Usage

### Requirements

```bash
pip install torch torchvision numpy matplotlib
```

### Run the notebook

```python
# Cell 1 — Load and visualise data
import pickle
from project_utils import ProjectDataset

with open('ocr_insurance_dataset.pkl', 'rb') as f:
    dataset = pickle.load(f)

show_dataset_images(dataset, num_images=5)
```

```python
# Cell 2 — Define and verify the model
model = OCRModel()

dummy_img  = torch.zeros(4, 1, 64, 64)
dummy_type = torch.zeros(4, 5)
output     = model(dummy_img, dummy_type)

print(output.shape)   # torch.Size([4, 2]) ✅
```

```python
# Cell 3 — Train
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    for images, types, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(images, types)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

---

## Key Concepts

| Term | Description |
|---|---|
| **Multi-modal** | Model that accepts inputs from more than one data type |
| **One-hot encoding** | Vector with a single `1` representing a category; all others `0` |
| **Conv2d** | 2D convolutional layer — slides learnable filters over an image |
| **padding=1** | Zero-border that preserves spatial size when kernel_size=3 |
| **MaxPool2d** | Downsampling via max value in each window — halves spatial dims |
| **ReLU** | Activation: `max(0, x)` — introduces non-linearity |
| **Flatten** | Reshapes `(B, C, H, W)` → `(B, C×H×W)` for linear layers |
| **Late fusion** | Combine modalities *after* separate processing, before classifier |
| **Dropout** | Randomly zeroes neurons during training to reduce overfitting |
| **Logits** | Raw model outputs before softmax — two values, one per class |

---

## Results

Training for **10 epochs** using:
- **Optimiser:** `Adam` (lr=1e-3)
- **Loss:** `CrossEntropyLoss`
- **Batch size:** configurable via `DataLoader`

---

## 💡 Tips & Gotchas

- Always run a **dummy forward pass** after defining the model — catches shape errors before training
- `img_tensor[0]` = image, `img_tensor[1]` = one-hot type — they are packed together per sample
- `torch.cat(dim=1)` = join features, not batches
- `Dropout` only activates during `model.train()` — remember to call `model.eval()` at inference
- The image sequential **must** be named `image_layer` — naming matters for checkpointing

---

*Built as part of the DigiNsure document digitisation initiative.*
