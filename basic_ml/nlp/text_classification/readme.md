# Text Classification Tutorial

This folder contains basic NLP tutorials for text classification, covering from simple Bag-of-Words to deep learning approaches.

## Overview

Text classification is a fundamental NLP task where we assign predefined categories to text documents. Common applications include:
- Sentiment analysis (positive/negative)
- Spam detection
- Topic classification
- Language identification

## Files

1. **`text_classification_basic.py`**: Simple Bag-of-Words approach with logistic regression
2. **`text_classification_tfidf.py`**: TF-IDF feature extraction with logistic regression
3. **`text_classification_pytorch.py`**: Deep learning approaches using CNN and RNN

## Concepts Explained

### 1. Bag-of-Words (BoW)

**Concept**: Represent text as a vector of word counts, ignoring word order.

**Example**:
```
Text: "I love machine learning"
Vocabulary: ["I", "love", "machine", "learning", "hate"]
Vector: [1, 1, 1, 1, 0]
```

**Advantages**:
- Simple and intuitive
- Fast to compute
- Works well for many tasks

**Limitations**:
- Loses word order information
- Ignores semantic relationships
- High-dimensional sparse vectors

### 2. TF-IDF (Term Frequency-Inverse Document Frequency)

**Concept**: Weight words by their importance:
- **TF (Term Frequency)**: How often a word appears in a document
- **IDF (Inverse Document Frequency)**: How rare a word is across all documents

**Formula**:
$$TF\text{-}IDF(t,d) = TF(t,d) \times IDF(t)$$

$$IDF(t) = \log\frac{N}{DF(t)}$$

where:
- $N$: Total number of documents
- $DF(t)$: Number of documents containing term $t$

**Advantages**:
- Reduces importance of common words (like "the", "a")
- Highlights important words
- Better than simple word counts

### 3. Text Preprocessing

Common preprocessing steps:
1. **Lowercasing**: Convert to lowercase
2. **Tokenization**: Split text into words
3. **Removing punctuation**: Clean special characters
4. **Stop word removal**: Remove common words (optional)
5. **Stemming/Lemmatization**: Reduce words to root form (optional)

### 4. Neural Network Approaches

#### CNN for Text (TextCNN)
- Uses 1D convolutions over word embeddings
- Captures local patterns (n-grams)
- Multiple filter sizes capture different n-gram patterns
- Fast and effective for many tasks

#### RNN for Text
- Processes text sequentially
- Captures long-term dependencies
- Can use LSTM or GRU for better memory
- Slower than CNN but better for long sequences

## Usage Examples

### Basic Bag-of-Words

```python
from text_classification_basic import SimpleTextClassifier

texts = ["I love this", "I hate this"]
labels = [1, 0]  # positive, negative

classifier = SimpleTextClassifier(max_features=100)
classifier.fit(texts, labels)
predictions = classifier.predict(["I like this"])
```

### TF-IDF

```python
from text_classification_tfidf import TFIDFTextClassifier

classifier = TFIDFTextClassifier(max_features=100)
classifier.fit(texts, labels)
predictions = classifier.predict(test_texts)
```

### PyTorch CNN/RNN

```python
from text_classification_pytorch import TextCNN, build_vocabulary, TextDataset
from torch.utils.data import DataLoader

# Build vocabulary
word_to_index = build_vocabulary(texts)

# Create dataset
dataset = TextDataset(texts, labels, word_to_index)
dataloader = DataLoader(dataset, batch_size=32)

# Create model
model = TextCNN(vocab_size=len(word_to_index), num_classes=2)
```

## Evaluation Metrics

- **Accuracy**: Overall correctness
- **Precision**: Of predicted positives, how many are actually positive
- **Recall**: Of actual positives, how many were predicted correctly
- **F1-Score**: Harmonic mean of precision and recall

## Next Steps

1. Try different preprocessing techniques
2. Experiment with different models
3. Use pre-trained word embeddings (Word2Vec, GloVe)
4. Try transformer models (BERT, GPT)
5. Handle imbalanced datasets

## Resources

- [scikit-learn Text Feature Extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
- [PyTorch NLP Tutorial](https://pytorch.org/tutorials/beginner/nlp/)
- [Text Classification Papers](https://paperswithcode.com/task/text-classification)

