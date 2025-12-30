"""
Text Classification using PyTorch and Neural Networks
More advanced deep learning approach
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import re

class TextDataset(Dataset):
    """Dataset class for text data"""
    
    def __init__(self, texts, labels, word_to_index, max_length=100):
        self.texts = texts
        self.labels = labels
        self.word_to_index = word_to_index
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Convert text to sequence of indices
        words = self._preprocess_text(text)
        sequence = [self.word_to_index.get(word, 0) for word in words[:self.max_length]]
        
        # Pad or truncate to max_length
        if len(sequence) < self.max_length:
            sequence += [0] * (self.max_length - len(sequence))
        else:
            sequence = sequence[:self.max_length]
        
        return torch.tensor(sequence, dtype=torch.long), torch.tensor(label, dtype=torch.long)
    
    def _preprocess_text(self, text):
        """Preprocess text"""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text.split()


class TextCNN(nn.Module):
    """Simple CNN for text classification"""
    
    def __init__(self, vocab_size, embedding_dim=100, num_classes=2, num_filters=100, filter_sizes=[3, 4, 5]):
        """
        Initialize TextCNN
        
        Parameters:
            vocab_size: Vocabulary size
            embedding_dim: Embedding dimension
            num_classes: Number of classes
            num_filters: Number of filters per filter size
            filter_sizes: List of filter sizes
        """
        super(TextCNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Convolutional layers for different filter sizes
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layer
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
    
    def forward(self, x):
        """
        Forward pass
        
        Parameters:
            x: Input tensor (batch_size, seq_length)
        
        Returns:
            Output logits (batch_size, num_classes)
        """
        # Embedding: (batch_size, seq_length) -> (batch_size, seq_length, embedding_dim)
        x = self.embedding(x)
        
        # Conv1d expects (batch_size, channels, seq_length)
        x = x.transpose(1, 2)  # (batch_size, embedding_dim, seq_length)
        
        # Apply convolutions and max pooling
        conv_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(x))  # (batch_size, num_filters, conv_seq_length)
            pooled = torch.max_pool1d(conv_out, kernel_size=conv_out.size(2))  # (batch_size, num_filters, 1)
            conv_outputs.append(pooled.squeeze(2))  # (batch_size, num_filters)
        
        # Concatenate all filter outputs
        x = torch.cat(conv_outputs, dim=1)  # (batch_size, len(filter_sizes) * num_filters)
        
        # Dropout and fully connected
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class SimpleRNN(nn.Module):
    """Simple RNN for text classification"""
    
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=128, num_classes=2, num_layers=1):
        """
        Initialize SimpleRNN
        
        Parameters:
            vocab_size: Vocabulary size
            embedding_dim: Embedding dimension
            hidden_dim: Hidden dimension
            num_classes: Number of classes
            num_layers: Number of RNN layers
        """
        super(SimpleRNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        """Forward pass"""
        # Embedding
        x = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        
        # RNN
        out, hidden = self.rnn(x)  # out: (batch_size, seq_length, hidden_dim)
        
        # Use the last hidden state
        last_hidden = out[:, -1, :]  # (batch_size, hidden_dim)
        
        # Fully connected
        output = self.fc(last_hidden)  # (batch_size, num_classes)
        
        return output


def build_vocabulary(texts, max_vocab_size=5000):
    """Build vocabulary from texts"""
    word_counts = Counter()
    
    for text in texts:
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        words = text.split()
        word_counts.update(words)
    
    # Select top N words
    most_common = word_counts.most_common(max_vocab_size)
    word_to_index = {word: idx + 1 for idx, (word, _) in enumerate(most_common)}  # 0 reserved for padding
    word_to_index['<UNK>'] = 0  # Unknown words
    
    return word_to_index


def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    """Train the model"""
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_texts, batch_labels in train_loader:
            batch_texts = batch_texts.to(device)
            batch_labels = batch_labels.to(device)
            
            # Forward pass
            outputs = model(batch_texts)
            loss = criterion(outputs, batch_labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
        
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')


def evaluate_model(model, test_loader, device):
    """Evaluate the model"""
    model.eval()
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for batch_texts, batch_labels in test_loader:
            batch_texts = batch_texts.to(device)
            outputs = model(batch_texts)
            _, predicted = torch.max(outputs.data, 1)
            
            all_labels.extend(batch_labels.numpy())
            all_predictions.extend(predicted.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_predictions)
    return accuracy, all_labels, all_predictions


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
    ] * 10  # Repeat to have more data
    
    labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 10
    
    # Build vocabulary
    word_to_index = build_vocabulary(texts, max_vocab_size=1000)
    vocab_size = len(word_to_index)
    
    print(f"Vocabulary size: {vocab_size}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    # Create datasets
    train_dataset = TextDataset(X_train, y_train, word_to_index, max_length=50)
    test_dataset = TextDataset(X_test, y_test, word_to_index, max_length=50)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Choose model: TextCNN or SimpleRNN
    model = TextCNN(vocab_size=vocab_size, embedding_dim=50, num_classes=2).to(device)
    # model = SimpleRNN(vocab_size=vocab_size, embedding_dim=50, hidden_dim=64, num_classes=2).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train
    print("Training model...")
    train_model(model, train_loader, criterion, optimizer, device, num_epochs=20)
    
    # Evaluate
    print("\nEvaluating model...")
    accuracy, y_true, y_pred = evaluate_model(model, test_loader, device)
    print(f"Test Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

