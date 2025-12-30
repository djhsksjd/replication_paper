import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import os

class LogisticRegressionEnhanced:
    """Enhanced Logistic Regression with regularization, advanced optimizers, and cross-validation"""
    
    def __init__(self, learning_rate=0.01, n_iters=1000, penalty='l2', 
                 l1_strength=0.01, l2_strength=0.01, optimizer='adam',
                 momentum=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 multi_class='binary'):
        """
        Initialize enhanced logistic regression model
        
        Parameters:
            learning_rate: Learning rate
            n_iters: Number of iterations
            penalty: Regularization type ('none', 'l1', 'l2', 'elasticnet')
            l1_strength: L1 regularization strength
            l2_strength: L2 regularization strength
            optimizer: Optimizer ('sgd', 'momentum', 'adam', 'rmsprop')
            momentum: Momentum parameter (for momentum optimizer)
            beta1: Adam optimizer first moment decay rate
            beta2: Adam optimizer second moment decay rate
            epsilon: Numerical stability parameter
            multi_class: 'binary' or 'multinomial'
        """
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.penalty = penalty.lower()
        self.l1_strength = l1_strength
        self.l2_strength = l2_strength
        self.optimizer = optimizer.lower()
        self.momentum = momentum
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.multi_class = multi_class
        
        self.weights = None
        self.bias = None
        self.loss_history = []
        self.n_classes = None
        
        # Validate parameters
        if self.penalty not in ['none', 'l1', 'l2', 'elasticnet']:
            raise ValueError("penalty must be 'none', 'l1', 'l2', or 'elasticnet'")
        if self.optimizer not in ['sgd', 'momentum', 'adam', 'rmsprop']:
            raise ValueError("optimizer must be 'sgd', 'momentum', 'adam', or 'rmsprop'")
    
    def _sigmoid(self, z):
        """Sigmoid activation function"""
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _softmax(self, z):
        """Softmax activation function"""
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def _one_hot_encode(self, y):
        """One-hot encode labels"""
        n_samples = len(y)
        y_onehot = np.zeros((n_samples, self.n_classes))
        y_onehot[np.arange(n_samples), y.astype(int)] = 1
        return y_onehot
    
    def fit(self, X, y):
        """
        Train the enhanced logistic regression model
        
        Parameters:
            X: Feature data (n_samples, n_features)
            y: Target labels (n_samples,)
        """
        n_samples, n_features = X.shape
        
        # Determine number of classes
        unique_classes = np.unique(y)
        self.n_classes = len(unique_classes)
        
        if self.multi_class == 'binary' and self.n_classes > 2:
            raise ValueError("Binary classification requires exactly 2 classes")
        
        # Initialize weights and bias
        if self.multi_class == 'binary':
            self.weights = np.zeros(n_features)
            self.bias = 0
        else:
            self.weights = np.zeros((n_features, self.n_classes))
            self.bias = np.zeros(self.n_classes)
        
        # Initialize optimizer states
        if self.optimizer == 'momentum':
            v_w = np.zeros_like(self.weights)
            v_b = np.zeros_like(self.bias) if isinstance(self.bias, np.ndarray) else 0
        elif self.optimizer == 'adam':
            m_w = np.zeros_like(self.weights)
            v_w = np.zeros_like(self.weights)
            m_b = np.zeros_like(self.bias) if isinstance(self.bias, np.ndarray) else 0
            v_b = np.zeros_like(self.bias) if isinstance(self.bias, np.ndarray) else 0
            t = 0
        elif self.optimizer == 'rmsprop':
            cache_w = np.zeros_like(self.weights)
            cache_b = np.zeros_like(self.bias) if isinstance(self.bias, np.ndarray) else 0
        
        # Training loop
        for i in range(self.n_iters):
            # Forward propagation
            if self.multi_class == 'binary':
                y_pred = self._predict_proba(X)
                loss = self._compute_loss_binary(y, y_pred)
                
                # Compute gradients
                dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
                db = (1 / n_samples) * np.sum(y_pred - y)
                
                # Add regularization gradients
                if self.penalty == 'l1' or self.penalty == 'elasticnet':
                    dw += self.l1_strength * np.sign(self.weights)
                if self.penalty == 'l2' or self.penalty == 'elasticnet':
                    dw += 2 * self.l2_strength * self.weights
                
                # Update parameters based on optimizer
                if self.optimizer == 'sgd':
                    self.weights -= self.learning_rate * dw
                    self.bias -= self.learning_rate * db
                elif self.optimizer == 'momentum':
                    v_w = self.momentum * v_w + self.learning_rate * dw
                    v_b = self.momentum * v_b + self.learning_rate * db
                    self.weights -= v_w
                    self.bias -= v_b
                elif self.optimizer == 'adam':
                    t += 1
                    m_w = self.beta1 * m_w + (1 - self.beta1) * dw
                    v_w = self.beta2 * v_w + (1 - self.beta2) * (dw ** 2)
                    m_b = self.beta1 * m_b + (1 - self.beta1) * db
                    v_b = self.beta2 * v_b + (1 - self.beta2) * (db ** 2)
                    
                    # Bias correction
                    m_w_hat = m_w / (1 - self.beta1 ** t)
                    v_w_hat = v_w / (1 - self.beta2 ** t)
                    m_b_hat = m_b / (1 - self.beta1 ** t)
                    v_b_hat = v_b / (1 - self.beta2 ** t)
                    
                    self.weights -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
                    self.bias -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
                elif self.optimizer == 'rmsprop':
                    cache_w = self.beta1 * cache_w + (1 - self.beta1) * (dw ** 2)
                    cache_b = self.beta1 * cache_b + (1 - self.beta1) * (db ** 2)
                    
                    self.weights -= self.learning_rate * dw / (np.sqrt(cache_w) + self.epsilon)
                    self.bias -= self.learning_rate * db / (np.sqrt(cache_b) + self.epsilon)
            else:
                # Multi-class classification
                y_pred = self._predict_proba(X)
                loss = self._compute_loss_multiclass(y, y_pred)
                
                y_onehot = self._one_hot_encode(y)
                
                # Compute gradients
                dw = (1 / n_samples) * np.dot(X.T, (y_pred - y_onehot))
                db = (1 / n_samples) * np.sum(y_pred - y_onehot, axis=0)
                
                # Add regularization gradients
                if self.penalty == 'l1' or self.penalty == 'elasticnet':
                    dw += self.l1_strength * np.sign(self.weights)
                if self.penalty == 'l2' or self.penalty == 'elasticnet':
                    dw += 2 * self.l2_strength * self.weights
                
                # Update parameters (similar to binary case but for each class)
                if self.optimizer == 'sgd':
                    self.weights -= self.learning_rate * dw
                    self.bias -= self.learning_rate * db
                elif self.optimizer == 'momentum':
                    if i == 0:
                        v_w = np.zeros_like(self.weights)
                        v_b = np.zeros_like(self.bias)
                    v_w = self.momentum * v_w + self.learning_rate * dw
                    v_b = self.momentum * v_b + self.learning_rate * db
                    self.weights -= v_w
                    self.bias -= v_b
                elif self.optimizer == 'adam':
                    if i == 0:
                        m_w = np.zeros_like(self.weights)
                        v_w = np.zeros_like(self.weights)
                        m_b = np.zeros_like(self.bias)
                        v_b = np.zeros_like(self.bias)
                        t = 0
                    t += 1
                    m_w = self.beta1 * m_w + (1 - self.beta1) * dw
                    v_w = self.beta2 * v_w + (1 - self.beta2) * (dw ** 2)
                    m_b = self.beta1 * m_b + (1 - self.beta1) * db
                    v_b = self.beta2 * v_b + (1 - self.beta2) * (db ** 2)
                    
                    m_w_hat = m_w / (1 - self.beta1 ** t)
                    v_w_hat = v_w / (1 - self.beta2 ** t)
                    m_b_hat = m_b / (1 - self.beta1 ** t)
                    v_b_hat = v_b / (1 - self.beta2 ** t)
                    
                    self.weights -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
                    self.bias -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
                elif self.optimizer == 'rmsprop':
                    if i == 0:
                        cache_w = np.zeros_like(self.weights)
                        cache_b = np.zeros_like(self.bias)
                    cache_w = self.beta1 * cache_w + (1 - self.beta1) * (dw ** 2)
                    cache_b = self.beta1 * cache_b + (1 - self.beta1) * (db ** 2)
                    
                    self.weights -= self.learning_rate * dw / (np.sqrt(cache_w) + self.epsilon)
                    self.bias -= self.learning_rate * db / (np.sqrt(cache_b) + self.epsilon)
            
            self.loss_history.append(loss)
            
            if (i + 1) % 100 == 0:
                print(f"Iteration {i+1}/{self.n_iters}, Loss: {loss:.4f}")
    
    def _predict_proba(self, X):
        """Predict class probabilities"""
        if self.multi_class == 'binary':
            z = np.dot(X, self.weights) + self.bias
            return self._sigmoid(z)
        else:
            z = np.dot(X, self.weights) + self.bias
            return self._softmax(z)
    
    def predict_proba(self, X):
        """Predict class probabilities for new data"""
        return self._predict_proba(X)
    
    def predict(self, X):
        """Predict class labels"""
        if self.multi_class == 'binary':
            probabilities = self._predict_proba(X)
            return (probabilities >= 0.5).astype(int)
        else:
            probabilities = self._predict_proba(X)
            return np.argmax(probabilities, axis=1)
    
    def _compute_loss_binary(self, y, y_pred):
        """Compute binary cross-entropy loss with regularization"""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        
        # Add regularization
        if self.penalty == 'l1':
            loss += self.l1_strength * np.sum(np.abs(self.weights))
        elif self.penalty == 'l2':
            loss += self.l2_strength * np.sum(self.weights ** 2)
        elif self.penalty == 'elasticnet':
            loss += self.l1_strength * np.sum(np.abs(self.weights)) + \
                   self.l2_strength * np.sum(self.weights ** 2)
        
        return loss
    
    def _compute_loss_multiclass(self, y, y_pred):
        """Compute multi-class cross-entropy loss with regularization"""
        y_onehot = self._one_hot_encode(y)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(np.sum(y_onehot * np.log(y_pred), axis=1))
        
        # Add regularization
        if self.penalty == 'l1':
            loss += self.l1_strength * np.sum(np.abs(self.weights))
        elif self.penalty == 'l2':
            loss += self.l2_strength * np.sum(self.weights ** 2)
        elif self.penalty == 'elasticnet':
            loss += self.l1_strength * np.sum(np.abs(self.weights)) + \
                   self.l2_strength * np.sum(self.weights ** 2)
        
        return loss
    
    def evaluate(self, X, y):
        """Evaluate model performance"""
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred)
        cm = confusion_matrix(y, y_pred)
        return accuracy, report, cm
    
    def cross_validate(self, X, y, n_splits=5):
        """Perform k-fold cross-validation"""
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        accuracy_scores = []
        
        print(f"Performing {n_splits}-fold cross-validation...")
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create new model instance
            fold_model = LogisticRegressionEnhanced(
                learning_rate=self.learning_rate,
                n_iters=self.n_iters,
                penalty=self.penalty,
                l1_strength=self.l1_strength,
                l2_strength=self.l2_strength,
                optimizer=self.optimizer,
                momentum=self.momentum,
                beta1=self.beta1,
                beta2=self.beta2,
                epsilon=self.epsilon,
                multi_class=self.multi_class
            )
            
            fold_model.fit(X_train, y_train)
            accuracy, _, _ = fold_model.evaluate(X_val, y_val)
            accuracy_scores.append(accuracy)
            
            print(f"Fold {fold+1}: Accuracy = {accuracy:.4f}")
        
        print(f"Cross-validation results - Mean Accuracy: {np.mean(accuracy_scores):.4f} ± {np.std(accuracy_scores):.4f}")
        
        return accuracy_scores
    
    def plot_loss(self):
        """Plot loss function"""
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.loss_history)), self.loss_history)
        plt.title('Loss vs. Iterations (Enhanced)')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig('loss_plot_logistic_enhanced.png')
        plt.show()
        print("Loss plot saved as loss_plot_logistic_enhanced.png")
    
    def plot_decision_boundary(self, X, y, resolution=100):
        """Plot decision boundary (2D features only)"""
        if X.shape[1] != 2:
            print("Warning: Decision boundary plot only works for 2D features")
            return
        
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                           np.linspace(y_min, y_max, resolution))
        
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        Z = self.predict(mesh_points)
        Z = Z.reshape(xx.shape)
        
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, levels=self.n_classes-1 if self.n_classes > 2 else 1,
                    alpha=0.5, cmap='viridis')
        plt.colorbar(label='Class')
        
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='black')
        plt.colorbar(scatter, label='True Label')
        plt.title('Decision Boundary (Enhanced)')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.grid(True, alpha=0.3)
        plt.savefig('decision_boundary_logistic_enhanced.png')
        plt.show()
        print("Decision boundary plot saved as decision_boundary_logistic_enhanced.png")

# Example usage
if __name__ == "__main__":
    if os.path.exists('data/X.npy') and os.path.exists('data/y.npy'):
        X = np.load('data/X.npy')
        y = np.load('data/y.npy')
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        n_classes = len(np.unique(y))
        multi_class = 'binary' if n_classes == 2 else 'multinomial'
        
        model = LogisticRegressionEnhanced(
            learning_rate=0.01,
            n_iters=1000,
            penalty='l2',
            l2_strength=0.01,
            optimizer='adam',
            multi_class=multi_class
        )
        model.fit(X_train, y_train)
        
        accuracy, report, cm = model.evaluate(X_test, y_test)
        print(f"\nTest Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report)
        
        model.cross_validate(X_train, y_train, n_splits=5)
        model.plot_loss()
        
        if X.shape[1] == 2:
            model.plot_decision_boundary(X_test, y_test)
    else:
        print("Data files not found. Please run data_generator.py first")

