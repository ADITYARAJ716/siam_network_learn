# Siamese Network on MNIST

A PyTorch implementation of a Siamese Network trained on MNIST to learn whether two handwritten digit images represent the **same digit** or **different digits**, using contrastive loss and embedding-space learning.

## Overview

Unlike a standard classifier that maps an image to a fixed set of classes, a Siamese network learns a **similarity metric**. Two images are passed through the same CNN (shared weights), producing two embedding vectors. The network is trained so that embeddings of the same digit end up close together, and embeddings of different digits end up far apart.

This project covers the full pipeline: pair dataset construction, model + loss definition, training, evaluation, embedding visualization (t-SNE), intermediate feature map inspection (PyTorch hooks), and inference on a saved model.

## Requirements

```
torch
torchvision
numpy
matplotlib
scikit-learn
```

Install with:
```bash
pip install torch torchvision numpy matplotlib scikit-learn
```

## Dataset

Uses `torchvision.datasets.MNIST`, downloaded automatically on first run. No manual setup needed.

## Project Structure / Pipeline

1. **Data loading** — Load MNIST train/test splits via `torchvision`.
2. **Pair dataset (`SiameseDataset`)** — Wraps MNIST to emit `(img1, img2, label)` triplets, where `label = 1` if both images are the same digit, `0` otherwise. Pairs are sampled randomly per epoch.
3. **Model (`SiameseNetwork`)** — A small shared CNN (`Conv → ReLU → MaxPool`, twice) followed by fully connected layers, producing a 64-dimensional embedding per image. Both images in a pair go through the *same* weights (`forward_one`).
4. **Loss (`ContrastiveLoss`)** — Pulls embeddings of matching pairs together, pushes non-matching pairs apart (up to a margin), using Euclidean distance between embeddings.
5. **Training loop** — Standard PyTorch loop (forward → loss → backward → step), tracking loss per epoch.
6. **Evaluation** — Computes pairwise distances on the test set, applies a distance threshold to predict same/different, and reports accuracy. A histogram visualizes how well same-digit vs. different-digit distances separate.
7. **t-SNE visualization** — Projects the 64D embeddings of ~1000 test images down to 2D to visually confirm that digits form distinct clusters.
8. **PyTorch hooks** — Inspects intermediate feature maps (Conv1, Conv2 outputs) for a sample image, to see what each layer is detecting.
9. **Save / load / inference** — Saves model weights (`state_dict`), reloads them, and runs inference on a random pair of test images.

## Model Architecture

```
Input: [1, 28, 28] grayscale image
  Conv2d(1 → 32, 3x3) → ReLU → MaxPool(2)   → [32, 14, 14]
  Conv2d(32 → 64, 3x3) → ReLU → MaxPool(2)  → [64, 7, 7]
  Flatten → Linear(3136 → 128) → ReLU → Linear(128 → 64)
Output: 64-dim embedding
```

## Usage

Run the script/notebook top to bottom. Key steps:

**Train:**
```python
model = SiameseNetwork().to(device)
# ... training loop ...
```

**Save:**
```python
torch.save(model.state_dict(), "siamese_mnist.pth")
```

**Load & run inference:**
```python
loaded_model = SiameseNetwork().to(device)
loaded_model.load_state_dict(torch.load("siamese_mnist.pth"))
loaded_model.eval()
```

**Visualize embeddings (t-SNE):**
```python
emb = model.forward_one(image)
# collect embeddings across many images, then TSNE().fit_transform(...)
```

**Inspect intermediate features (hooks):**
```python
hook = MultiFeatureHook({'Conv1': model.cnn[0], 'Conv2': model.cnn[3]})
_ = model.forward_one(sample_image)
plot_multi_hook_features(hook)
hook.remove()
```

## Outputs / Artifacts

- `siamese_mnist.pth` — saved model weights
- Training loss curve (matplotlib)
- Distance histogram (same vs. different pairs)
- t-SNE scatter plot of embeddings, colored by digit
- Feature map grids from hooked Conv layers

## Key Concepts Used

- Siamese (twin) networks with shared weights
- Contrastive loss and margin-based distance learning
- Embedding space evaluation via distance thresholding
- t-SNE for high-dimensional embedding visualization
- PyTorch forward hooks for model interpretability

## Notes

- Trained and tested on an RTX 4050 (6GB VRAM); MNIST's small image size keeps memory usage minimal.
- Distance threshold (default `0.5`) can be tuned — try a few values and compare accuracy.
- `perplexity` in t-SNE (default `30`) significantly affects cluster appearance; experiment with different values.
