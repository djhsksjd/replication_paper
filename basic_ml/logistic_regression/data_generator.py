import numpy as np
import os

class LogisticRegressionDataGenerator:
    """Logistic Regression Data Generator for Binary and Multi-class Classification"""
    
    def __init__(self, n_samples=1000, n_features=2, n_classes=2, noise_level=0.1, random_state=42):
        """
        Initialize data generator
        
        Parameters:
            n_samples: Number of samples
            n_features: Number of features
            n_classes: Number of classes (2 for binary, >2 for multi-class)
            noise_level: Noise level (standard deviation of noise)
            random_state: Random seed
        """
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_classes = n_classes
        self.noise_level = noise_level
        self.random_state = random_state
    
    def generate_binary_data(self, centers=None, class_sep=1.0):
        """
        Generate binary classification data
        
        Parameters:
            centers: Class centers, if None then randomly generated
            class_sep: Separation between classes
        
        Returns:
            X: Feature data (n_samples, n_features)
            y: Target labels (n_samples,) with values 0 or 1
        """
        np.random.seed(self.random_state)
        
        if centers is None:
            # Generate two class centers
            centers = [
                np.random.randn(self.n_features) * class_sep,
                np.random.randn(self.n_features) * class_sep + np.ones(self.n_features) * class_sep
            ]
        
        # Generate samples for each class
        n_samples_per_class = self.n_samples // 2
        X = []
        y = []
        
        for class_idx in range(2):
            class_samples = np.random.randn(n_samples_per_class, self.n_features) * self.noise_level
            class_samples += centers[class_idx]
            X.append(class_samples)
            y.append(np.full(n_samples_per_class, class_idx))
        
        X = np.vstack(X)
        y = np.hstack(y)
        
        # Shuffle the data
        indices = np.random.permutation(self.n_samples)
        X = X[indices]
        y = y[indices]
        
        return X, y
    
    def generate_multiclass_data(self, centers=None, class_sep=1.0):
        """
        Generate multi-class classification data
        
        Parameters:
            centers: Class centers, if None then randomly generated
            class_sep: Separation between classes
        
        Returns:
            X: Feature data (n_samples, n_features)
            y: Target labels (n_samples,) with values 0 to n_classes-1
        """
        np.random.seed(self.random_state)
        
        if centers is None:
            # Generate class centers in a circle
            centers = []
            for i in range(self.n_classes):
                angle = 2 * np.pi * i / self.n_classes
                center = np.array([np.cos(angle), np.sin(angle)]) * class_sep
                if self.n_features > 2:
                    center = np.pad(center, (0, self.n_features - 2), mode='constant')
                centers.append(center)
        
        # Generate samples for each class
        n_samples_per_class = self.n_samples // self.n_classes
        X = []
        y = []
        
        for class_idx in range(self.n_classes):
            class_samples = np.random.randn(n_samples_per_class, self.n_features) * self.noise_level
            class_samples += centers[class_idx]
            X.append(class_samples)
            y.append(np.full(n_samples_per_class, class_idx))
        
        X = np.vstack(X)
        y = np.hstack(y)
        
        # Shuffle the data
        indices = np.random.permutation(self.n_samples)
        X = X[indices]
        y = y[indices]
        
        return X, y
    
    def generate_data(self, centers=None, class_sep=1.0):
        """
        Generate classification data (binary or multi-class)
        
        Parameters:
            centers: Class centers, if None then randomly generated
            class_sep: Separation between classes
        
        Returns:
            X: Feature data (n_samples, n_features)
            y: Target labels (n_samples,)
        """
        if self.n_classes == 2:
            return self.generate_binary_data(centers, class_sep)
        else:
            return self.generate_multiclass_data(centers, class_sep)
    
    def save_data(self, X, y, directory='data'):
        """
        Save data to files
        
        Parameters:
            X: Feature data
            y: Target labels
            directory: Save directory
        """
        # Create directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)
        
        # Save data
        np.save(os.path.join(directory, 'X.npy'), X)
        np.save(os.path.join(directory, 'y.npy'), y)
        
        print(f"Data saved to {directory} directory")
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        print(f"Class distribution: {np.bincount(y.astype(int))}")

# Example usage
if __name__ == "__main__":
    # Binary classification example
    print("Generating binary classification data...")
    generator_binary = LogisticRegressionDataGenerator(
        n_samples=1000, 
        n_features=2, 
        n_classes=2, 
        noise_level=0.5
    )
    X_binary, y_binary = generator_binary.generate_data()
    generator_binary.save_data(X_binary, y_binary, directory='data/binary')
    
    # Multi-class classification example
    print("\nGenerating multi-class classification data...")
    generator_multiclass = LogisticRegressionDataGenerator(
        n_samples=1500, 
        n_features=2, 
        n_classes=3, 
        noise_level=0.5
    )
    X_multiclass, y_multiclass = generator_multiclass.generate_data()
    generator_multiclass.save_data(X_multiclass, y_multiclass, directory='data/multiclass')

