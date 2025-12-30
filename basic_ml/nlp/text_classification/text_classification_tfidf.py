"""
Text Classification using TF-IDF and Machine Learning
More advanced approach than Bag-of-Words
"""

import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import math

class TFIDFTextClassifier:
    """Text classifier using TF-IDF features"""
    
    def __init__(self, max_features=1000):
        """
        Initialize TF-IDF classifier
        
        Parameters:
            max_features: Maximum number of features
        """
        self.max_features = max_features
        self.vocabulary = {}
        self.word_to_index = {}
        self.idf = None
        self.n_documents = 0
    
    def preprocess_text(self, text):
        """Preprocess text"""
        import re
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text.split()
    
    def build_vocabulary(self, texts):
        """Build vocabulary and compute IDF"""
        word_counts = Counter()
        word_doc_freq = Counter()  # Document frequency
        
        for text in texts:
            words = set(self.preprocess_text(text))
            word_counts.update(words)
            word_doc_freq.update(words)
        
        # Select top N words
        most_common = word_counts.most_common(self.max_features)
        self.word_to_index = {word: idx for idx, (word, _) in enumerate(most_common)}
        self.vocabulary = set(self.word_to_index.keys())
        
        # Compute IDF
        self.n_documents = len(texts)
        self.idf = {}
        for word in self.vocabulary:
            # Document frequency (how many documents contain this word)
            doc_freq = word_doc_freq.get(word, 0)
            # IDF = log(total_docs / docs_with_word)
            self.idf[word] = math.log(self.n_documents / (doc_freq + 1))
        
        print(f"Vocabulary size: {len(self.vocabulary)}")
    
    def compute_tfidf(self, text):
        """
        Compute TF-IDF vector for a text
        
        Parameters:
            text: Input text
        
        Returns:
            TF-IDF vector
        """
        words = self.preprocess_text(text)
        vector = np.zeros(self.max_features)
        
        # Term Frequency (TF)
        word_counts = Counter(words)
        total_words = len(words)
        
        for word, count in word_counts.items():
            if word in self.word_to_index:
                idx = self.word_to_index[word]
                # TF = count / total_words
                tf = count / total_words if total_words > 0 else 0
                # TF-IDF = TF * IDF
                vector[idx] = tf * self.idf.get(word, 0)
        
        return vector
    
    def fit(self, X_train, y_train, learning_rate=0.01, n_iters=1000):
        """Train classifier"""
        # Build vocabulary and IDF
        self.build_vocabulary(X_train)
        
        # Convert to TF-IDF vectors
        X_vectors = np.array([self.compute_tfidf(text) for text in X_train])
        y_train = np.array(y_train)
        
        # Initialize weights
        n_features = X_vectors.shape[1]
        n_classes = len(np.unique(y_train))
        
        if n_classes == 2:
            self.weights = np.zeros(n_features)
            self.bias = 0
        else:
            self.weights = np.zeros((n_features, n_classes))
            self.bias = np.zeros(n_classes)
        
        self.n_classes = n_classes
        
        # Training
        for i in range(n_iters):
            if n_classes == 2:
                z = np.dot(X_vectors, self.weights) + self.bias
                y_pred_proba = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
                
                dw = np.dot(X_vectors.T, (y_pred_proba - y_train)) / len(X_train)
                db = np.mean(y_pred_proba - y_train)
                
                self.weights -= learning_rate * dw
                self.bias -= learning_rate * db
            else:
                z = np.dot(X_vectors, self.weights) + self.bias
                exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
                y_pred_proba = exp_z / np.sum(exp_z, axis=1, keepdims=True)
                
                y_onehot = np.zeros((len(y_train), n_classes))
                y_onehot[np.arange(len(y_train)), y_train] = 1
                
                dw = np.dot(X_vectors.T, (y_pred_proba - y_onehot)) / len(X_train)
                db = np.mean(y_pred_proba - y_onehot, axis=0)
                
                self.weights -= learning_rate * dw
                self.bias -= learning_rate * db
            
            if (i + 1) % 100 == 0:
                accuracy = accuracy_score(y_train, self.predict(X_train))
                print(f"Iteration {i+1}/{n_iters}, Accuracy: {accuracy:.4f}")
    
    def predict(self, texts):
        """Predict labels"""
        X_vectors = np.array([self.compute_tfidf(text) for text in texts])
        
        if self.n_classes == 2:
            z = np.dot(X_vectors, self.weights) + self.bias
            proba = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
            return (proba >= 0.5).astype(int)
        else:
            z = np.dot(X_vectors, self.weights) + self.bias
            exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
            return np.argmax(exp_z / np.sum(exp_z, axis=1, keepdims=True), axis=1)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model"""
        predictions = self.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)
        return accuracy, report


# Example usage
if __name__ == "__main__":
    # Sample data
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
    
    labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
    
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.3, random_state=42
    )
    
    classifier = TFIDFTextClassifier(max_features=100)
    classifier.fit(X_train, y_train, learning_rate=0.1, n_iters=500)
    
    accuracy, report = classifier.evaluate(X_test, y_test)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)

