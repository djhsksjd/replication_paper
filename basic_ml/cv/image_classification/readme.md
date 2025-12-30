# Image Classification with CNNs

This folder contains tutorials for image classification using Convolutional Neural Networks (CNNs).

## Overview

Image classification is a fundamental computer vision task where we assign labels to images. CNNs are the standard architecture for this task.

## Files

1. **`cnn_basic.py`**: Basic CNN architectures (SimpleCNN, LeNet)
2. **`cnn_advanced.py`**: Advanced architectures (ResNet, VGG)

## CNN Concepts Explained

### 1. Convolutional Layer

**Purpose**: Extract local features from images

**Key Parameters**:
- **Kernel/Filter size**: Size of the convolution window (e.g., 3x3, 5x5)
- **Stride**: How much the filter moves (e.g., stride=1 moves 1 pixel)
- **Padding**: Add zeros around image to preserve size
- **Number of filters**: How many different features to detect

**Example**:
```python
# 3x3 convolution with 32 filters
conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
```

### 2. Pooling Layer

**Purpose**: Reduce spatial dimensions, reduce parameters

**Types**:
- **Max Pooling**: Takes maximum value in window
- **Average Pooling**: Takes average value in window

**Example**:
```python
# 2x2 max pooling with stride 2
pool = nn.MaxPool2d(kernel_size=2, stride=2)
# Reduces 32x32 -> 16x16
```

### 3. Activation Functions

- **ReLU**: Most common, introduces non-linearity
- **Sigmoid/Tanh**: Less common in CNNs

### 4. Fully Connected Layer

**Purpose**: Final classification based on extracted features

### 5. Batch Normalization

**Purpose**: Normalize activations, speeds up training, improves stability

### 6. Dropout

**Purpose**: Prevent overfitting by randomly setting some neurons to zero

## Architecture Patterns

### Basic CNN Structure

```
Input Image (3, 32, 32)
    ↓
Conv + ReLU + Pool
    ↓
Conv + ReLU + Pool
    ↓
Conv + ReLU + Pool
    ↓
Flatten
    ↓
FC Layer
    ↓
Output (num_classes)
```

### ResNet (Residual Networks)

**Key Innovation**: Skip connections (residual connections)

**Benefits**:
- Easier to train deep networks
- Helps with vanishing gradient problem
- Can learn identity mappings

**Residual Block**:
```
Input
  ↓
Conv → BN → ReLU → Conv → BN
  ↓                      ↓
  └──────── Add ─────────┘
            ↓
          ReLU
            ↓
         Output
```

### VGG Architecture

**Key Features**:
- Deep stacks of 3x3 convolutions
- Multiple pooling layers
- Simple and effective

## Training Tips

1. **Data Augmentation**: Rotate, flip, crop images to increase dataset size
2. **Learning Rate**: Start with 0.001, use learning rate scheduling
3. **Batch Size**: Typically 32-128 depending on GPU memory
4. **Regularization**: Use dropout and batch normalization
5. **Transfer Learning**: Use pre-trained models for better performance

## Common Datasets

- **CIFAR-10**: 10 classes, 32x32 images, 50K training images
- **CIFAR-100**: 100 classes, 32x32 images
- **ImageNet**: 1000 classes, large-scale dataset
- **MNIST**: Handwritten digits, 28x28 grayscale

## Usage Example

```python
from cnn_basic import SimpleCNN, train_model, evaluate_model
import torch.optim as optim

# Create model
model = SimpleCNN(num_classes=10)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train
train_losses, train_accuracies = train_model(
    model, train_loader, criterion, optimizer, device, num_epochs=10
)

# Evaluate
test_accuracy = evaluate_model(model, test_loader, device)
```

## Next Steps

1. Experiment with different architectures
2. Try transfer learning with pre-trained models
3. Implement data augmentation
4. Tune hyperparameters
5. Try attention mechanisms
6. Explore object detection and segmentation

## Resources

- [CS231n: Convolutional Neural Networks](http://cs231n.github.io/convolutional-networks/)
- [PyTorch Vision Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)

