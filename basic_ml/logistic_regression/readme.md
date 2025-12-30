# Logistic Regression Algorithm Implementation

This folder contains Python and PyTorch implementations of the Logistic Regression algorithm, along with a data generator for classification problems.

## Algorithm Overview

Logistic Regression is a fundamental **supervised learning algorithm** for **classification tasks**. Unlike linear regression which predicts continuous values, logistic regression predicts **class probabilities** and assigns samples to discrete classes.

### Core Concept

Logistic Regression models the probability that a sample belongs to a particular class using the **logistic (sigmoid) function**:

$$P(y=1|\mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$

where:
- $\mathbf{x}$: Feature vector
- $\mathbf{w}$: Weight vector
- $b$: Bias term
- $\sigma$: Sigmoid function

## Folder Structure

- `data_generator.py`: Data generator for binary and multi-class classification problems
- `logistic_regression_python.py`: Pure Python implementation using NumPy
- `logistic_regression_pytorch.py`: PyTorch implementation
- `logistic_regression_enhanced.py`: Enhanced version with regularization, advanced optimizers, and cross-validation
- `readme.md`: This documentation file

## Environment Requirements

- Python 3.6+
- numpy
- scikit-learn (for evaluation metrics)
- matplotlib (for visualization)
- PyTorch (for PyTorch implementation)

## Usage Steps

### 1. Generate Data

First, run the data generator to create training and test data:

```bash
python data_generator.py
```

This will generate:
- Binary classification data in `data/binary/`
- Multi-class classification data in `data/multiclass/`

### 2. Run Python Implementation

```bash
python logistic_regression_python.py
```

### 3. Run PyTorch Implementation

```bash
python logistic_regression_pytorch.py
```

### 4. Run Enhanced Implementation

```bash
python logistic_regression_enhanced.py
```

## Mathematical Foundation

### 1. Binary Classification

For binary classification (two classes: 0 and 1), logistic regression uses the **sigmoid function** to map linear combinations to probabilities:

**Sigmoid Function:**
$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**Properties:**
- Output range: (0, 1)
- S-shaped curve
- $\sigma(0) = 0.5$ (decision boundary)

**Decision Rule:**
- If $P(y=1|\mathbf{x}) \geq 0.5$: Predict class 1
- If $P(y=1|\mathbf{x}) < 0.5$: Predict class 0

### 2. Multi-class Classification

For multi-class problems (K classes), logistic regression extends to **multinomial logistic regression** using the **softmax function**:

**Softmax Function:**
$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

**Properties:**
- Outputs a probability distribution over K classes
- $\sum_{i=1}^{K} \text{softmax}(z_i) = 1$
- Each output is in [0, 1]

**Decision Rule:**
- Predict the class with the highest probability: $\arg\max_i P(y=i|\mathbf{x})$

### 3. Loss Function: Cross-Entropy

Logistic regression uses **cross-entropy loss** (also called log loss) instead of MSE:

**Binary Cross-Entropy:**
$$\mathcal{L} = -\frac{1}{n}\sum_{i=1}^{n}[y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$$

**Multi-class Cross-Entropy:**
$$\mathcal{L} = -\frac{1}{n}\sum_{i=1}^{n}\sum_{k=1}^{K}y_{ik}\log(\hat{y}_{ik})$$

where:
- $y_i$: True label (binary) or one-hot encoded (multi-class)
- $\hat{y}_i$: Predicted probability
- $n$: Number of samples

**Why Cross-Entropy?**
- Penalizes confident wrong predictions heavily
- Provides better gradients for optimization
- Aligns with maximum likelihood estimation

### 4. Optimization: Gradient Descent

The gradient of cross-entropy loss with respect to weights:

**Binary Classification:**
$$\frac{\partial \mathcal{L}}{\partial \mathbf{w}} = \frac{1}{n}\sum_{i=1}^{n}(\hat{y}_i - y_i)\mathbf{x}_i$$

**Update Rule:**
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \cdot \frac{\partial \mathcal{L}}{\partial \mathbf{w}_t}$$

where $\eta$ is the learning rate.

### 5. Regularization

To prevent overfitting, regularization terms can be added:

**L2 Regularization (Ridge):**
$$\mathcal{L}_{\text{ridge}} = \mathcal{L} + \lambda \|\mathbf{w}\|_2^2$$

**L1 Regularization (Lasso):**
$$\mathcal{L}_{\text{lasso}} = \mathcal{L} + \lambda \|\mathbf{w}\|_1$$

**Elastic Net:**
$$\mathcal{L}_{\text{elastic}} = \mathcal{L} + \lambda_1 \|\mathbf{w}\|_1 + \lambda_2 \|\mathbf{w}\|_2^2$$

## Key Differences from Linear Regression

| Aspect | Linear Regression | Logistic Regression |
|--------|------------------|---------------------|
| **Task** | Regression (continuous output) | Classification (discrete output) |
| **Output** | Real numbers | Probabilities [0, 1] |
| **Activation** | None (linear) | Sigmoid/Softmax |
| **Loss Function** | MSE | Cross-Entropy |
| **Decision Boundary** | Continuous line/plane | S-shaped curve (sigmoid) |

## Implementation Features

### Python Implementation (`logistic_regression_python.py`)
- Pure NumPy implementation
- Supports binary and multi-class classification
- Gradient descent optimization
- Visualization of decision boundaries

### PyTorch Implementation (`logistic_regression_pytorch.py`)
- PyTorch-based implementation
- Automatic differentiation
- Batch processing support
- Model saving/loading

### Enhanced Implementation (`logistic_regression_enhanced.py`)
- **Regularization**: L1, L2, and Elastic Net
- **Advanced Optimizers**: SGD, Momentum, Adam, RMSprop
- **Cross-Validation**: K-fold cross-validation
- **Hyperparameter Tuning**: Grid search support

## Model Evaluation Metrics

### Classification Metrics

1. **Accuracy**: Proportion of correct predictions
   $$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

2. **Precision**: Proportion of positive predictions that are correct
   $$Precision = \frac{TP}{TP + FP}$$

3. **Recall**: Proportion of actual positives that are correctly identified
   $$Recall = \frac{TP}{TP + FN}$$

4. **F1-Score**: Harmonic mean of precision and recall
   $$F1 = 2 \cdot \frac{Precision \cdot Recall}{Precision + Recall}$$

5. **Confusion Matrix**: Shows true vs predicted class distribution

## Visualization Outputs

- `loss_plot_logistic.png`: Loss function over iterations
- `decision_boundary_logistic.png`: Decision boundary visualization (2D features only)

## Parameter Tuning Guide

### Learning Rate
- **Too high**: Loss may diverge or oscillate
- **Too low**: Slow convergence
- **Recommended**: Start with 0.01, adjust based on loss curve

### Regularization Strength
- **L2 (λ)**: Typically 0.001 to 1.0
- **L1 (λ)**: Typically 0.001 to 0.1
- Use cross-validation to find optimal values

### Number of Iterations
- Monitor loss curve to determine convergence
- Early stopping can prevent overfitting

## Advantages and Limitations

### Advantages
- ✅ Simple and interpretable
- ✅ Fast training and prediction
- ✅ Provides probability estimates
- ✅ No feature scaling required (though recommended)
- ✅ Works well with small datasets

### Limitations
- ❌ Assumes linear decision boundary
- ❌ Sensitive to outliers
- ❌ May underperform on complex non-linear problems
- ❌ Requires feature independence assumption

## Extensions and Variants

1. **Polynomial Logistic Regression**: Add polynomial features for non-linear boundaries
2. **Regularized Logistic Regression**: L1/L2 regularization for feature selection/overfitting prevention
3. **Multinomial Logistic Regression**: Extend to multi-class problems
4. **Ordinal Logistic Regression**: For ordinal target variables

## Example Use Cases

- **Medical Diagnosis**: Predict disease presence (binary) or disease type (multi-class)
- **Email Spam Detection**: Classify emails as spam or not spam
- **Image Classification**: Classify images into categories
- **Sentiment Analysis**: Classify text as positive/negative/neutral
- **Credit Risk Assessment**: Predict loan default probability

## Further Reading

- Maximum Likelihood Estimation (MLE) for logistic regression
- Generalized Linear Models (GLM)
- Regularization techniques in classification
- Multi-class classification strategies (One-vs-Rest, One-vs-One)

## Code Examples

### Basic Usage

```python
from logistic_regression_python import LogisticRegressionPython
import numpy as np

# Generate or load data
X_train, y_train = ...  # Training data
X_test, y_test = ...    # Test data

# Create and train model
model = LogisticRegressionPython(learning_rate=0.01, n_iters=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
probabilities = model.predict_proba(X_test)

# Evaluate
accuracy, report, cm = model.evaluate(X_test, y_test)
```

### Enhanced Usage with Regularization

```python
from logistic_regression_enhanced import LogisticRegressionEnhanced

# Create model with L2 regularization and Adam optimizer
model = LogisticRegressionEnhanced(
    learning_rate=0.01,
    n_iters=1000,
    penalty='l2',
    l2_strength=0.01,
    optimizer='adam'
)

model.fit(X_train, y_train)

# Cross-validation
cv_scores = model.cross_validate(X_train, y_train, n_splits=5)
```

---

**Note**: This implementation is designed for educational purposes. For production use, consider using optimized libraries like scikit-learn or PyTorch's built-in modules.

