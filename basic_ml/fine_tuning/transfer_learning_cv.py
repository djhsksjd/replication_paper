"""
Transfer Learning for Computer Vision
Fine-tuning pre-trained models for image classification
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from torchvision import models
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class TransferLearningModel:
    """Transfer learning wrapper for image classification"""
    
    def __init__(self, model_name='resnet18', num_classes=10, pretrained=True):
        """
        Initialize transfer learning model
        
        Parameters:
            model_name: Model name ('resnet18', 'resnet34', 'vgg16', 'alexnet')
            num_classes: Number of output classes
            pretrained: Use pre-trained weights
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load pre-trained model
        if model_name == 'resnet18':
            self.model = models.resnet18(pretrained=pretrained)
            # Replace final layer
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, num_classes)
        elif model_name == 'resnet34':
            self.model = models.resnet34(pretrained=pretrained)
            num_features = self.model.fc.in_features
            self.model.fc = nn.Linear(num_features, num_classes)
        elif model_name == 'vgg16':
            self.model = models.vgg16(pretrained=pretrained)
            num_features = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_features, num_classes)
        elif model_name == 'alexnet':
            self.model = models.alexnet(pretrained=pretrained)
            num_features = self.model.classifier[6].in_features
            self.model.classifier[6] = nn.Linear(num_features, num_classes)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.model.to(self.device)
        print(f"Using device: {self.device}")
        print(f"Model: {model_name} (pretrained={pretrained})")
    
    def freeze_backbone(self):
        """Freeze all layers except the final classifier"""
        if 'resnet' in self.model_name:
            for param in self.model.parameters():
                param.requires_grad = False
            # Unfreeze final layer
            for param in self.model.fc.parameters():
                param.requires_grad = True
        elif self.model_name == 'vgg16':
            for param in self.model.features.parameters():
                param.requires_grad = False
            for param in self.model.classifier.parameters():
                param.requires_grad = True
        print("Backbone frozen, only classifier trainable")
    
    def unfreeze_all(self):
        """Unfreeze all layers"""
        for param in self.model.parameters():
            param.requires_grad = True
        print("All layers trainable")
    
    def train(self, train_loader, val_loader, num_epochs=10, learning_rate=0.001, 
              freeze_backbone_epochs=0):
        """
        Train the model
        
        Parameters:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of epochs
            learning_rate: Learning rate
            freeze_backbone_epochs: Number of epochs to freeze backbone
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        train_losses = []
        train_accuracies = []
        val_accuracies = []
        
        # Freeze backbone for initial epochs
        if freeze_backbone_epochs > 0:
            self.freeze_backbone()
        
        for epoch in range(num_epochs):
            # Unfreeze after freeze_backbone_epochs
            if epoch == freeze_backbone_epochs:
                self.unfreeze_all()
            
            # Training phase
            self.model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
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
            
            train_loss = running_loss / len(train_loader)
            train_acc = 100 * correct / total
            train_losses.append(train_loss)
            train_accuracies.append(train_acc)
            
            # Validation phase
            val_acc = self.evaluate(val_loader)
            val_accuracies.append(val_acc)
            
            scheduler.step()
            
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {train_loss:.4f}, '
                  f'Train Acc: {train_acc:.2f}%, '
                  f'Val Acc: {val_acc:.2f}%')
        
        return train_losses, train_accuracies, val_accuracies
    
    def evaluate(self, data_loader):
        """Evaluate model"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in data_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        return accuracy
    
    def save_model(self, path):
        """Save model"""
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load model"""
        self.model.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")


def get_data_loaders(dataset_name='cifar10', batch_size=32):
    """
    Get data loaders for common datasets
    
    Parameters:
        dataset_name: 'cifar10' or 'cifar100'
        batch_size: Batch size
    
    Returns:
        train_loader, test_loader
    """
    # Data augmentation for training
    transform_train = transforms.Compose([
        transforms.Resize(224),  # Resize for ImageNet pre-trained models
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(224, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet stats
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    if dataset_name == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train
        )
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test
        )
        num_classes = 10
    elif dataset_name == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform_train
        )
        testset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform_test
        )
        num_classes = 100
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader, num_classes


# Example usage
if __name__ == "__main__":
    # Get data
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader, num_classes = get_data_loaders('cifar10', batch_size=32)
    
    # Create model
    model = TransferLearningModel(
        model_name='resnet18',
        num_classes=num_classes,
        pretrained=True
    )
    
    # Train with transfer learning
    print("\nTraining with transfer learning...")
    train_losses, train_accs, val_accs = model.train(
        train_loader, test_loader,
        num_epochs=5,
        learning_rate=0.001,
        freeze_backbone_epochs=2  # Freeze for 2 epochs, then fine-tune all
    )
    
    # Final evaluation
    final_acc = model.evaluate(test_loader)
    print(f"\nFinal Test Accuracy: {final_acc:.2f}%")
    
    # Save model
    model.save_model('transfer_learning_model.pth')
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train')
    plt.plot(val_accs, label='Val')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('transfer_learning_history.png')
    plt.show()

