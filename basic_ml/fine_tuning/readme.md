# Fine-tuning and Transfer Learning Tutorials

This folder contains tutorials for fine-tuning pre-trained models in NLP and Computer Vision.

## Overview

Fine-tuning (transfer learning) is a technique where we take a pre-trained model and adapt it to a new task. This is much more efficient than training from scratch.

## Files

1. **`bert_finetuning.py`**: Fine-tuning BERT for text classification
2. **`transfer_learning_cv.py`**: Transfer learning for image classification

## Concepts Explained

### 1. Transfer Learning

**Concept**: Use knowledge learned from one task to help with a different but related task.

**Why it works**:
- Pre-trained models have learned useful features
- Lower layers capture general patterns (edges, textures, syntax)
- Only need to adapt higher layers for specific tasks

**Benefits**:
- ✅ Faster training
- ✅ Better performance with less data
- ✅ Lower computational cost

### 2. Fine-tuning Strategies

#### Strategy 1: Freeze Backbone, Train Classifier
- Freeze all pre-trained layers
- Only train the final classification layer
- Fastest, works well when dataset is small

#### Strategy 2: Gradual Unfreezing
- Start by training only classifier
- Gradually unfreeze layers from top to bottom
- Good balance between speed and performance

#### Strategy 3: Full Fine-tuning
- Unfreeze all layers
- Train entire model with lower learning rate
- Best performance, requires more data and computation

### 3. Learning Rate Considerations

**Key Points**:
- Use **lower learning rate** than training from scratch
- Pre-trained weights are already good, need gentle updates
- Typical: 1e-5 to 1e-3 (vs 1e-3 to 1e-1 for training from scratch)

**Learning Rate Scheduling**:
- Start with low LR
- Gradually decrease (step decay, cosine annealing)
- Helps fine-tune more precisely

### 4. BERT Fine-tuning

**BERT (Bidirectional Encoder Representations from Transformers)**:
- Pre-trained on large text corpus
- Understands context and semantics
- Can be fine-tuned for various NLP tasks

**Fine-tuning Process**:
1. Load pre-trained BERT model
2. Add task-specific head (classification layer)
3. Train on your dataset with low learning rate
4. Usually only need 2-4 epochs

**Key Parameters**:
- `learning_rate`: 2e-5 to 5e-5 (very low!)
- `max_length`: 128 or 256 tokens
- `batch_size`: 16 or 32
- `num_epochs`: 2-4 (often enough)

### 5. Transfer Learning for Vision

**Pre-trained Models**:
- **ResNet**: Deep residual networks, very popular
- **VGG**: Simple and effective
- **AlexNet**: Classic architecture
- All trained on ImageNet (1.2M images, 1000 classes)

**Process**:
1. Load pre-trained model (trained on ImageNet)
2. Replace final classification layer
3. Optionally freeze early layers
4. Train on your dataset

**Data Preprocessing**:
- Use ImageNet normalization stats
- Resize to 224x224 (ImageNet size)
- Apply data augmentation

## Usage Examples

### BERT Fine-tuning

```python
from bert_finetuning import BERTFineTuner

# Initialize
fine_tuner = BERTFineTuner(
    model_name='bert-base-uncased',
    num_classes=2
)

# Prepare data
train_loader, val_loader = fine_tuner.prepare_data(texts, labels)

# Fine-tune
fine_tuner.train(train_loader, val_loader, num_epochs=3)

# Predict
predictions = fine_tuner.predict(test_texts)
```

### Transfer Learning for CV

```python
from transfer_learning_cv import TransferLearningModel

# Create model
model = TransferLearningModel(
    model_name='resnet18',
    num_classes=10,
    pretrained=True
)

# Freeze backbone initially
model.freeze_backbone()

# Train
model.train(train_loader, val_loader, num_epochs=10)

# Unfreeze and fine-tune all
model.unfreeze_all()
model.train(train_loader, val_loader, num_epochs=5)
```

## Best Practices

### 1. When to Use Transfer Learning

✅ **Use when**:
- Limited training data
- Similar task to pre-training
- Want faster development
- Need good baseline quickly

❌ **Avoid when**:
- Very different domain
- Have massive dataset
- Need custom architecture

### 2. Hyperparameter Tips

- **Learning Rate**: Start with 1e-5, adjust based on loss
- **Batch Size**: As large as GPU allows
- **Epochs**: Usually 2-10 epochs sufficient
- **Freezing**: Freeze for 1-2 epochs, then unfreeze

### 3. Common Pitfalls

- **Learning rate too high**: Can destroy pre-trained weights
- **Too few epochs**: Underfit the adaptation
- **Too many epochs**: Overfit to your small dataset
- **Wrong preprocessing**: Must match pre-training preprocessing

## Comparison: Training from Scratch vs Fine-tuning

| Aspect | From Scratch | Fine-tuning |
|--------|--------------|-------------|
| **Data needed** | Large (10K+) | Small (100+) |
| **Training time** | Days/weeks | Hours |
| **Performance** | Good (with enough data) | Excellent (even with little data) |
| **Computational cost** | High | Low |
| **Best for** | Novel tasks | Similar tasks |

## Advanced Techniques

1. **Progressive Unfreezing**: Unfreeze layers gradually
2. **Differential Learning Rates**: Different LR for different layers
3. **Knowledge Distillation**: Transfer knowledge to smaller model
4. **Multi-task Learning**: Fine-tune on multiple tasks

## Resources

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [PyTorch Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)

