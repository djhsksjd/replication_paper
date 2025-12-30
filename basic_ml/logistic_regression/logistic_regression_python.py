import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os

class LogisticRegressionPython:
    """Logistic Regression implementation using pure Python and NumPy"""
    
    def __init__(self, learning_rate=0.01, n_iters=1000, multi_class='binary'):
        """
        Initialize Logistic Regression model
        
        Parameters:
            learning_rate: Learning rate for gradient descent
            n_iters: Number of iterations
            multi_class: 'binary' for binary classification, 'multinomial' for multi-class
        """
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.multi_class = multi_class
        self.weights = None
        self.bias = None
        self.loss_history = []
        self.n_classes = None
    
    def _sigmoid(self, z):
        """
        Sigmoid activation function
        
        Parameters:
            z: Input value
        
        Returns:
            Sigmoid of z
        """
        # Clip z to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _softmax(self, z):
        """
        Softmax activation function for multi-class classification
        
        Parameters:
            z: Input values (n_samples, n_classes)
        
        Returns:
            Softmax probabilities (n_samples, n_classes)
        """
        # Subtract max for numerical stability
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    def fit(self, X, y):
        """
        Train the logistic regression model
        
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
            # For multi-class, we need weights for each class
            self.weights = np.zeros((n_features, self.n_classes))
            self.bias = np.zeros(self.n_classes)
        
        # Gradient descent
        for i in range(self.n_iters):
            # Forward propagation
            if self.multi_class == 'binary':
                y_pred = self._predict_proba(X)
                loss = self._compute_loss_binary(y, y_pred)
                
                # Compute gradients
                dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
                db = (1 / n_samples) * np.sum(y_pred - y)
                
                # Update parameters
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
            else:
                # Multi-class classification
                y_pred = self._predict_proba(X)
                loss = self._compute_loss_multiclass(y, y_pred)
                
                # One-hot encode labels
                y_onehot = self._one_hot_encode(y)
                
                # Compute gradients
                dw = (1 / n_samples) * np.dot(X.T, (y_pred - y_onehot))
                db = (1 / n_samples) * np.sum(y_pred - y_onehot, axis=0)
                
                # Update parameters
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
            
            self.loss_history.append(loss)
            
            # Print progress every 100 iterations
            if (i + 1) % 100 == 0:
                print(f"Iteration {i+1}/{self.n_iters}, Loss: {loss:.4f}")
    
    def _predict_proba(self, X):
        """
        Predict class probabilities
        
        Parameters:
            X: Feature data
        
        Returns:
            Predicted probabilities
        """
        if self.multi_class == 'binary':
            z = np.dot(X, self.weights) + self.bias
            return self._sigmoid(z)
        else:
            z = np.dot(X, self.weights) + self.bias
            return self._softmax(z)
    
    def predict_proba(self, X):
        """
        Predict class probabilities for new data
        
        Parameters:
            X: New feature data
        
        Returns:
            Predicted probabilities
        """
        return self._predict_proba(X)
    
    def predict(self, X):
        """
        Predict class labels for new data
        
        Parameters:
            X: New feature data
        
        Returns:
            Predicted class labels
        """
        if self.multi_class == 'binary':
            probabilities = self._predict_proba(X)
            return (probabilities >= 0.5).astype(int)
        else:
            probabilities = self._predict_proba(X)
            return np.argmax(probabilities, axis=1)
    
    def _compute_loss_binary(self, y, y_pred):
        """
        Compute binary cross-entropy loss
        
        Parameters:
            y: True labels
            y_pred: Predicted probabilities
        
        Returns:
            Loss value
        """
        # Avoid log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        return loss
    
    def _compute_loss_multiclass(self, y, y_pred):
        """
        Compute multi-class cross-entropy loss
        
        Parameters:
            y: True labels
            y_pred: Predicted probabilities
        
        Returns:
            Loss value
        """
        y_onehot = self._one_hot_encode(y)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(np.sum(y_onehot * np.log(y_pred), axis=1))
        return loss
    
    def _one_hot_encode(self, y):
        """
        One-hot encode labels
        
        Parameters:
            y: Labels
        
        Returns:
            One-hot encoded labels
        """
        n_samples = len(y)
        y_onehot = np.zeros((n_samples, self.n_classes))
        y_onehot[np.arange(n_samples), y.astype(int)] = 1
        return y_onehot
    
    def evaluate(self, X, y):
        """
        Evaluate model performance
        
        Parameters:
            X: Feature data
            y: True labels
        
        Returns:
            accuracy: Classification accuracy
            report: Classification report
            cm: Confusion matrix
        """
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred)
        cm = confusion_matrix(y, y_pred)
        return accuracy, report, cm
    
    def plot_loss(self):
        """
        Plot loss function over iterations
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.loss_history)), self.loss_history)
        plt.title('Loss vs. Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig('loss_plot_logistic.png')
        plt.show()
        print("Loss plot saved as loss_plot_logistic.png")
    
    def plot_decision_boundary(self, X, y, resolution=100):
        """
        Plot decision boundary (only for 2D features)
        
        Parameters:
            X: Feature data
            y: Labels
            resolution: Resolution of the decision boundary plot
        """
        if X.shape[1] != 2:
            print("Warning: Decision boundary plot only works for 2D features")
            return
        
        # Create a mesh
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution),
                           np.linspace(y_min, y_max, resolution))
        
        # Predict on mesh points
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        if self.multi_class == 'binary':
            Z = self._predict_proba(mesh_points)
            Z = Z.reshape(xx.shape)
        else:
            Z = self.predict(mesh_points)
            Z = Z.reshape(xx.shape)
        
        # Plot
        plt.figure(figsize=(10, 8))
        if self.multi_class == 'binary':
            plt.contourf(xx, yy, Z, levels=50, alpha=0.5, cmap='RdYlBu')
            plt.colorbar(label='Probability')
        else:
            plt.contourf(xx, yy, Z, levels=self.n_classes-1, alpha=0.5, cmap='viridis')
            plt.colorbar(label='Class')
        
        # Plot data points
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='black')
        plt.colorbar(scatter, label='True Label')
        plt.title('Decision Boundary')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.grid(True, alpha=0.3)
        plt.savefig('decision_boundary_logistic.png')
        plt.show()
        print("Decision boundary plot saved as decision_boundary_logistic.png")

# Example usage
if __name__ == "__main__":
    # Check if data files exist
    if os.path.exists('data/X.npy') and os.path.exists('data/y.npy'):
        # Load data
        X = np.load('data/X.npy')
        y = np.load('data/y.npy')
        
        # Split into train and test sets
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Determine if binary or multi-class
        n_classes = len(np.unique(y))
        multi_class = 'binary' if n_classes == 2 else 'multinomial'
        
        # Create and train model
        model = LogisticRegressionPython(
            learning_rate=0.01, 
            n_iters=1000,
            multi_class=multi_class
        )
        model.fit(X_train, y_train)
        
        # Evaluate model
        accuracy, report, cm = model.evaluate(X_test, y_test)
        print(f"\nTest Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Plot loss
        model.plot_loss()
        
        # Plot decision boundary if 2D features
        if X.shape[1] == 2:
            model.plot_decision_boundary(X_test, y_test)
    else:
        print("Data files not found. Please run data_generator.py first")

