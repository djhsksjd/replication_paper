"""
BERT Fine-tuning Tutorial
Fine-tuning pre-trained BERT for text classification
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

class TextDataset(Dataset):
    """Dataset for text classification"""
    
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class BERTFineTuner:
    """BERT fine-tuning for text classification"""
    
    def __init__(self, model_name='bert-base-uncased', num_classes=2, max_length=128):
        """
        Initialize BERT fine-tuner
        
        Parameters:
            model_name: Pre-trained BERT model name
            num_classes: Number of classification classes
            max_length: Maximum sequence length
        """
        self.model_name = model_name
        self.num_classes = num_classes
        self.max_length = max_length
        
        # Load tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes
        )
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        print(f"Using device: {self.device}")
        print(f"Model: {model_name}")
    
    def prepare_data(self, texts, labels, batch_size=16):
        """
        Prepare data loaders
        
        Parameters:
            texts: List of text strings
            labels: List of labels
            batch_size: Batch size
        
        Returns:
            train_loader, val_loader
        """
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        
        # Create datasets
        train_dataset = TextDataset(X_train, y_train, self.tokenizer, self.max_length)
        val_dataset = TextDataset(X_val, y_val, self.tokenizer, self.max_length)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader
    
    def train(self, train_loader, val_loader, num_epochs=3, learning_rate=2e-5):
        """
        Fine-tune BERT model
        
        Parameters:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            learning_rate: Learning rate
        """
        # Optimizer
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, correct_bias=False)
        
        # Learning rate scheduler
        total_steps = len(train_loader) * num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # Training loop
        for epoch in range(num_epochs):
            print(f'\nEpoch {epoch+1}/{num_epochs}')
            print('-' * 30)
            
            # Training phase
            self.model.train()
            train_loss = 0
            
            for batch in train_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                train_loss += loss.item()
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            avg_train_loss = train_loss / len(train_loader)
            
            # Validation phase
            val_loss, val_accuracy = self.evaluate(val_loader)
            
            print(f'Train Loss: {avg_train_loss:.4f}')
            print(f'Val Loss: {val_loss:.4f}')
            print(f'Val Accuracy: {val_accuracy:.4f}')
    
    def evaluate(self, data_loader):
        """
        Evaluate model
        
        Parameters:
            data_loader: Data loader
        
        Returns:
            Average loss and accuracy
        """
        self.model.eval()
        val_loss = 0
        predictions = []
        true_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                val_loss += outputs.loss.item()
                
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        avg_loss = val_loss / len(data_loader)
        accuracy = accuracy_score(true_labels, predictions)
        
        return avg_loss, accuracy
    
    def predict(self, texts, batch_size=16):
        """
        Make predictions on new texts
        
        Parameters:
            texts: List of text strings
            batch_size: Batch size
        
        Returns:
            Predictions
        """
        self.model.eval()
        predictions = []
        
        dataset = TextDataset(texts, [0] * len(texts), self.tokenizer, self.max_length)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                predictions.extend(preds.cpu().numpy())
        
        return predictions
    
    def save_model(self, path):
        """Save fine-tuned model"""
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load fine-tuned model"""
        self.model = BertForSequenceClassification.from_pretrained(path)
        self.tokenizer = BertTokenizer.from_pretrained(path)
        self.model.to(self.device)
        print(f"Model loaded from {path}")


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
    ] * 10  # Repeat for more data
    
    labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0] * 10
    
    # Initialize fine-tuner
    print("Initializing BERT fine-tuner...")
    fine_tuner = BERTFineTuner(
        model_name='bert-base-uncased',
        num_classes=2,
        max_length=128
    )
    
    # Prepare data
    train_loader, val_loader = fine_tuner.prepare_data(texts, labels, batch_size=4)
    
    # Fine-tune
    print("\nStarting fine-tuning...")
    fine_tuner.train(train_loader, val_loader, num_epochs=3, learning_rate=2e-5)
    
    # Test predictions
    test_texts = [
        "This is a wonderful film",
        "I hate this boring movie"
    ]
    predictions = fine_tuner.predict(test_texts)
    
    print("\nPredictions:")
    for text, pred in zip(test_texts, predictions):
        sentiment = "Positive" if pred == 1 else "Negative"
        print(f"'{text}' -> {sentiment}")
    
    # Save model
    fine_tuner.save_model('./bert_finetuned')

