"""
Basic CNN for Image Classification
Simple tutorial for beginners
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class SimpleCNN(nn.Module):
    """Simple Convolutional Neural Network for image classification"""
    
    def __init__(self, num_classes=10):
        """
        Initialize SimpleCNN
        
        Parameters:
            num_classes: Number of output classes
        """
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        # Input: (batch_size, 3, 32, 32) for RGB images
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        # After 3 pooling operations: 32/2/2/2 = 4, so 4x4x128 = 2048
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        """
        Forward pass
        
        Parameters:
            x: Input tensor (batch_size, 3, 32, 32)
        
        Returns:
            Output logits (batch_size, num_classes)
        """
        # Conv block 1: 32x32 -> 16x16
        x = self.pool(F.relu(self.conv1(x)))
        
        # Conv block 2: 16x16 -> 8x8
        x = self.pool(F.relu(self.conv2(x)))
        
        # Conv block 3: 8x8 -> 4x4
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten: (batch_size, 128, 4, 4) -> (batch_size, 2048)
        x = x.view(-1, 128 * 4 * 4)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


class LeNet(nn.Module):
    """LeNet-5 architecture (classic CNN)"""
    
    def __init__(self, num_classes=10):
        super(LeNet, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
        self.pool = nn.MaxPool2d(2, 2)
    
    def forward(self, x):
        # Conv + Pool: 32x32 -> 28x28 -> 14x14
        x = self.pool(F.relu(self.conv1(x)))
        
        # Conv + Pool: 14x14 -> 10x10 -> 5x5
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten
        x = x.view(-1, 16 * 5 * 5)
        
        # Fully connected
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x


def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    """Train the CNN model"""
    model.train()
    train_losses = []
    train_accuracies = []
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total
        
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')
    
    return train_losses, train_accuracies


def evaluate_model(model, test_loader, device):
    """Evaluate the CNN model"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy


def plot_training_history(train_losses, train_accuracies):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot loss
    ax1.plot(train_losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(train_accuracies)
    ax2.set_title('Training Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()


# Example usage with CIFAR-10
if __name__ == "__main__":
    import torchvision
    import torchvision.transforms as transforms
    
    # Data preprocessing
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Load CIFAR-10 dataset
    print("Loading CIFAR-10 dataset...")
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2
    )
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=False, num_workers=2
    )
    
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Choose model
    model = SimpleCNN(num_classes=10).to(device)
    # model = LeNet(num_classes=10).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train
    print("\nTraining model...")
    train_losses, train_accuracies = train_model(
        model, trainloader, criterion, optimizer, device, num_epochs=10
    )
    
    # Evaluate
    print("\nEvaluating model...")
    test_accuracy = evaluate_model(model, testloader, device)
    
    # Plot training history
    plot_training_history(train_losses, train_accuracies)
    
    # Save model
    torch.save(model.state_dict(), 'cnn_model.pth')
    print("Model saved as cnn_model.pth")

