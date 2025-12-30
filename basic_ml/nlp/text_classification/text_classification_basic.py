"""
Basic Text Classification using Bag-of-Words and Logistic Regression
This is a simple NLP tutorial for beginners
"""

import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re

class SimpleTextClassifier:
    """Simple text classifier using Bag-of-Words approach"""
    
    def __init__(self, max_features=1000):
        """
        Initialize text classifier
        
        Parameters:
            max_features: Maximum number of features (vocabulary size)
        """
        self.max_features = max_features
        self.vocabulary = {}
        self.word_to_index = {}
        self.model = None
    
    def preprocess_text(self, text):
        """
        Simple text preprocessing
        
        Parameters:
            text: Input text string
        
        Returns:
            List of words (tokens)
        """
        # Convert to lowercase
        text = text.lower()
        # Remove special characters, keep only alphanumeric and spaces
        text = re.sub(r'[^a-z0-9\s]', '', text)
        # Split into words
        words = text.split()
        return words
    
    def build_vocabulary(self, texts):
        """
        Build vocabulary from training texts
        
        Parameters:
            texts: List of text strings
        """
        # Count word frequencies
        word_counts = Counter()
        for text in texts:
            words = self.preprocess_text(text)
            word_counts.update(words)
        
        # Select top N most frequent words
        most_common = word_counts.most_common(self.max_features)
        self.word_to_index = {word: idx for idx, (word, _) in enumerate(most_common)}
        self.vocabulary = set(self.word_to_index.keys())
        print(f"Vocabulary size: {len(self.vocabulary)}")
    
    def text_to_vector(self, text):
        """
        Convert text to feature vector (Bag-of-Words)
        
        Parameters:
            text: Input text string
        
        Returns:
            Feature vector (numpy array)
        """
        words = self.preprocess_text(text)
        vector = np.zeros(self.max_features)
        
        for word in words:
            if word in self.word_to_index:
                idx = self.word_to_index[word]
                vector[idx] += 1
        
        # Normalize by text length (optional)
        if len(words) > 0:
            vector = vector / len(words)
        
        return vector
    
    def fit(self, X_train, y_train, learning_rate=0.01, n_iters=1000):
        """
        Train the classifier using logistic regression
        
        Parameters:
            X_train: List of training texts
            y_train: Training labels
            learning_rate: Learning rate
            n_iters: Number of iterations
        """
        # Build vocabulary
        self.build_vocabulary(X_train)
        
        # Convert texts to vectors
        X_vectors = np.array([self.text_to_vector(text) for text in X_train])
        y_train = np.array(y_train)
        
        # Initialize logistic regression weights
        n_features = X_vectors.shape[1]
        n_classes = len(np.unique(y_train))
        
        if n_classes == 2:
            # Binary classification
            self.weights = np.zeros(n_features)
            self.bias = 0
        else:
            # Multi-class classification
            self.weights = np.zeros((n_features, n_classes))
            self.bias = np.zeros(n_classes)
        
        self.n_classes = n_classes
        
        # Training loop
        for i in range(n_iters):
            if n_classes == 2:
                # Binary classification
                z = np.dot(X_vectors, self.weights) + self.bias
                y_pred_proba = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
                
                # Compute gradients
                dw = np.dot(X_vectors.T, (y_pred_proba - y_train)) / len(X_train)
                db = np.mean(y_pred_proba - y_train)
                
                # Update weights
                self.weights -= learning_rate * dw
                self.bias -= learning_rate * db
            else:
                # Multi-class classification
                z = np.dot(X_vectors, self.weights) + self.bias
                exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
                y_pred_proba = exp_z / np.sum(exp_z, axis=1, keepdims=True)
                
                # One-hot encode labels
                y_onehot = np.zeros((len(y_train), n_classes))
                y_onehot[np.arange(len(y_train)), y_train] = 1
                
                # Compute gradients
                dw = np.dot(X_vectors.T, (y_pred_proba - y_onehot)) / len(X_train)
                db = np.mean(y_pred_proba - y_onehot, axis=0)
                
                # Update weights
                self.weights -= learning_rate * dw
                self.bias -= learning_rate * db
            
            # Print progress
            if (i + 1) % 100 == 0:
                accuracy = self._compute_accuracy(X_train, y_train)
                print(f"Iteration {i+1}/{n_iters}, Training Accuracy: {accuracy:.4f}")
    
    def predict_proba(self, texts):
        """
        Predict class probabilities
        
        Parameters:
            texts: List of text strings
        
        Returns:
            Predicted probabilities
        """
        X_vectors = np.array([self.text_to_vector(text) for text in texts])
        
        if self.n_classes == 2:
            z = np.dot(X_vectors, self.weights) + self.bias
            proba = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
            return np.column_stack([1 - proba, proba])
        else:
            z = np.dot(X_vectors, self.weights) + self.bias
            exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
            return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def predict(self, texts):
        """
        Predict class labels
        
        Parameters:
            texts: List of text strings
        
        Returns:
            Predicted labels
        """
        proba = self.predict_proba(texts)
        if self.n_classes == 2:
            return (proba[:, 1] >= 0.5).astype(int)
        else:
            return np.argmax(proba, axis=1)
    
    def _compute_accuracy(self, texts, labels):
        """Compute accuracy"""
        predictions = self.predict(texts)
        return accuracy_score(labels, predictions)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Parameters:
            X_test: Test texts
            y_test: Test labels
        
        Returns:
            accuracy, classification_report, confusion_matrix
        """
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)
        cm = confusion_matrix(y_test, predictions)
        return accuracy, report, cm


# Example usage
if __name__ == "__main__":
    # Sample data: Simple sentiment analysis
    texts = [
        "I love this movie it is amazing",
        "This film is terrible and boring",
        "Great acting and wonderful story",
        "Worst movie I have ever seen",
        "Fantastic plot and excellent direction",
        "Boring and waste of time",
        "Amazing cinematography beautiful scenes",
        "Disappointing and poorly made",
        "Outstanding performance by actors",
        "Bad script and weak storyline"
    ]
    
    labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 = positive, 0 = negative
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.3, random_state=42
    )
    
    # Create and train classifier
    classifier = SimpleTextClassifier(max_features=100)
    classifier.fit(X_train, y_train, learning_rate=0.1, n_iters=500)
    
    # Evaluate
    accuracy, report, cm = classifier.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Test on new texts
    test_texts = [
        "This is a wonderful film",
        "I hate this boring movie"
    ]
    predictions = classifier.predict(test_texts)
    print(f"\nPredictions:")
    for text, pred in zip(test_texts, predictions):
        sentiment = "Positive" if pred == 1 else "Negative"
        print(f"'{text}' -> {sentiment}")

