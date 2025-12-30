import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader, TensorDataset

class LogisticRegressionPyTorch(nn.Module):
    """Logistic Regression implementation using PyTorch"""
    
    def __init__(self, input_dim, num_classes=2):
        """
        Initialize Logistic Regression model
        
        Parameters:
            input_dim: Input feature dimension
            num_classes: Number of classes (2 for binary, >2 for multi-class)
        """
        super(LogisticRegressionPyTorch, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        self.num_classes = num_classes
        self.loss_history = []
    
    def forward(self, x):
        """
        Forward pass
        
        Parameters:
            x: Input feature data
        
        Returns:
            Logits (raw scores before softmax)
        """
        return self.linear(x)
    
    def fit(self, X, y, learning_rate=0.01, n_iters=1000, batch_size=32):
        """
        Train the logistic regression model
        
        Parameters:
            X: Feature data (n_samples, n_features)
            y: Target labels (n_samples,)
            learning_rate: Learning rate
            n_iters: Number of iterations (epochs)
            batch_size: Batch size
        """
        # Convert to PyTorch tensors
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.long)
        
        # Create dataset and dataloader
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Define loss function and optimizer
        if self.num_classes == 2:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()
        
        optimizer = optim.SGD(self.parameters(), lr=learning_rate)
        
        # Training loop
        self.train()
        for epoch in range(n_iters):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                # Forward pass
                outputs = self(batch_X)
                
                # Compute loss
                if self.num_classes == 2:
                    # For binary classification, squeeze output and convert y to float
                    outputs = outputs.squeeze()
                    if outputs.dim() == 0:
                        outputs = outputs.unsqueeze(0)
                    batch_y = batch_y.float()
                    loss = criterion(outputs, batch_y)
                else:
                    loss = criterion(outputs, batch_y)
                
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item() * batch_X.size(0)
            
            # Average loss for the epoch
            epoch_loss /= len(dataset)
            self.loss_history.append(epoch_loss)
            
            # Print progress every 100 epochs
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{n_iters}, Loss: {epoch_loss:.4f}")
    
    def predict_proba(self, X):
        """
        Predict class probabilities
        
        Parameters:
            X: Feature data
        
        Returns:
            Predicted probabilities (numpy array)
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        
        self.eval()
        with torch.no_grad():
            outputs = self(X)
            if self.num_classes == 2:
                probabilities = torch.sigmoid(outputs)
                # Return probabilities for both classes
                prob_0 = 1 - probabilities
                prob_1 = probabilities
                return torch.stack([prob_0.squeeze(), prob_1.squeeze()], dim=1).numpy()
            else:
                probabilities = torch.softmax(outputs, dim=1)
                return probabilities.numpy()
    
    def predict(self, X):
        """
        Predict class labels
        
        Parameters:
            X: Feature data
        
        Returns:
            Predicted class labels (numpy array)
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        
        self.eval()
        with torch.no_grad():
            outputs = self(X)
            if self.num_classes == 2:
                probabilities = torch.sigmoid(outputs)
                predictions = (probabilities >= 0.5).long().squeeze()
            else:
                _, predictions = torch.max(outputs, 1)
            return predictions.numpy()
    
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
        Plot loss function over epochs
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.loss_history)), self.loss_history)
        plt.title('Loss vs. Epochs')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.grid(True)
        plt.savefig('loss_plot_logistic_pytorch.png')
        plt.show()
        print("Loss plot saved as loss_plot_logistic_pytorch.png")
    
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
        Z = self.predict(mesh_points)
        Z = Z.reshape(xx.shape)
        
        # Plot
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, levels=self.num_classes-1 if self.num_classes > 2 else 1, 
                    alpha=0.5, cmap='viridis')
        plt.colorbar(label='Class')
        
        # Plot data points
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='black')
        plt.colorbar(scatter, label='True Label')
        plt.title('Decision Boundary (PyTorch)')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.grid(True, alpha=0.3)
        plt.savefig('decision_boundary_logistic_pytorch.png')
        plt.show()
        print("Decision boundary plot saved as decision_boundary_logistic_pytorch.png")
    
    def save_model(self, path='logistic_regression_model.pth'):
        """
        Save model
        
        Parameters:
            path: Model save path
        """
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load_model(self, path='logistic_regression_model.pth'):
        """
        Load model
        
        Parameters:
            path: Model load path
        """
        self.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")

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
        
        # Determine number of classes
        num_classes = len(np.unique(y))
        
        # Create and train model
        input_dim = X.shape[1]
        model = LogisticRegressionPyTorch(input_dim, num_classes=num_classes)
        model.fit(X_train, y_train, learning_rate=0.01, n_iters=1000, batch_size=32)
        
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
        
        # Save model
        model.save_model()
    else:
        print("Data files not found. Please run data_generator.py first")

