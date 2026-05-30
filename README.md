# MNIST Digit Classification — MLP vs CNN

Comparing two neural network architectures on the MNIST handwritten digit dataset using PyTorch.

## Models

### `first.py` — Fully-Connected Network (MLP)
A 3-layer feedforward network that flattens each 28×28 image into a 784-dimensional vector.

| Layer | In → Out |
|-------|----------|
| Linear + ReLU | 784 → 128 |
| Linear + ReLU | 128 → 64 |
| Linear (logits) | 64 → 10 |

### `second.py` — Convolutional Neural Network (CNN)
A CNN that preserves spatial structure through two convolutional layers before classifying.

| Layer | Output shape |
|-------|-------------|
| Conv2d(1→32) + ReLU + MaxPool | (32, 14, 14) |
| Conv2d(32→64) + ReLU + MaxPool | (64, 7, 7) |
| Flatten | 3136 |
| Linear + ReLU + Dropout(0.25) | 128 |
| Linear (logits) | 10 |

## Results

Both models trained for 5 epochs with Adam (lr=0.001) and CrossEntropyLoss on an Apple MPS device.

| Model | Test Accuracy |
|-------|--------------|
| MLP   | ~97.5%       |
| CNN   | ~99.0%       |

The CNN achieves higher accuracy by exploiting the 2D spatial structure of images — convolutional filters detect edges and patterns regardless of their position in the image.

## Requirements

```
torch
torchvision
```

Install with:
```bash
pip install torch torchvision
```

## Usage

```bash
# Train and evaluate the MLP
python first.py

# Train and evaluate the CNN
python second.py
```

MNIST data is downloaded automatically to `./data/` on first run.
