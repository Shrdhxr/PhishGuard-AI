# trainer.py
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from tqdm import tqdm

# Import the EmailDataset and create_data_loaders from data_selector
from data_selector import EmailDataset, create_data_loaders

class PhishingClassifier(nn.Module):
    def __init__(self, num_classes, model_name='bert-base-uncased'):
        super(PhishingClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        return self.classifier(output)

class Trainer:
    def __init__(self, model, train_loader, val_loader, device, num_classes):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_classes = num_classes
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=2, gamma=0.1)
        
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc='Training')
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * correct/total:.2f}%'
            })
        
        return total_loss / len(self.train_loader), correct / total
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        return (total_loss / len(self.val_loader), correct / total, 
                all_predictions, all_labels)
    
    def train(self, epochs=10, save_path='models/'):
        os.makedirs(save_path, exist_ok=True)
        
        best_val_acc = 0
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')
            print('-' * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, val_preds, val_labels = self.validate()
            
            # Update scheduler
            self.scheduler.step()
            
            # Store metrics
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)
            
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss
                }, os.path.join(save_path, 'best_model.pth'))
                print(f'New best model saved with val_acc: {val_acc:.4f}')
        
        # Plot training history
        self.plot_training_history(train_losses, val_losses, train_accs, val_accs, save_path)
        
        return train_losses, val_losses, train_accs, val_accs
    
    def plot_training_history(self, train_losses, val_losses, train_accs, val_accs, save_path):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(train_losses, label='Train Loss')
        ax1.plot(val_losses, label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Accuracy plot
        ax2.plot(train_accs, label='Train Accuracy')
        ax2.plot(val_accs, label='Val Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'training_history.png'))
        plt.close()

def evaluate_model(model, test_loader, device, idx_to_type=None):
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Classification report
    target_names = [idx_to_type[i] for i in range(len(idx_to_type))] if idx_to_type else None
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=target_names))
    
    # Confusion matrix
    if idx_to_type:
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(all_labels, all_predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=target_names,
                    yticklabels=target_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('models/confusion_matrix.png')
        plt.close()
    
    return all_predictions, all_labels

def main():
    # Check device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load raw data
    with open('data/processed/binary_data.pkl', 'rb') as f:
        binary_data = pickle.load(f)
    
    with open('data/processed/multiclass_data.pkl', 'rb') as f:
        multiclass_data = pickle.load(f)
    
    # Create DataLoaders
    binary_loaders, multiclass_loaders = create_data_loaders(binary_data, multiclass_data, batch_size=8)
    
    # Train binary classifier
    print("Training Binary Classifier...")
    binary_model = PhishingClassifier(num_classes=binary_loaders['num_classes']).to(device)
    binary_trainer = Trainer(
        binary_model, 
        binary_loaders['train_loader'], 
        binary_loaders['val_loader'],
        device,
        binary_loaders['num_classes']
    )
    binary_trainer.train(epochs=5, save_path='models/binary/')
    
    # Evaluate binary model
    print("\nEvaluating Binary Model...")
    binary_model.load_state_dict(torch.load('models/binary/best_model.pth')['model_state_dict'])
    evaluate_model(binary_model, binary_loaders['test_loader'], device)
    
    # Train multiclass classifier
    print("\nTraining Multiclass Classifier...")
    multiclass_model = PhishingClassifier(num_classes=multiclass_loaders['num_classes']).to(device)
    multiclass_trainer = Trainer(
        multiclass_model,
        multiclass_loaders['train_loader'],
        multiclass_loaders['val_loader'],
        device,
        multiclass_loaders['num_classes']
    )
    multiclass_trainer.train(epochs=10, save_path='models/multiclass/')
    
    # Evaluate multiclass model
    print("\nEvaluating Multiclass Model...")
    multiclass_model.load_state_dict(torch.load('models/multiclass/best_model.pth')['model_state_dict'])
    evaluate_model(
        multiclass_model, 
        multiclass_loaders['test_loader'], 
        device, 
        multiclass_loaders.get('idx_to_type')
    )
    
    # Save final models and metadata
    torch.save(binary_model.state_dict(), 'models/binary_model_final.pth')
    torch.save(multiclass_model.state_dict(), 'models/multiclass_model_final.pth')
    
    # Save multiclass mapping
    with open('models/multiclass_mapping.pkl', 'wb') as f:
        pickle.dump({
            'type_to_idx': multiclass_loaders['type_to_idx'],
            'idx_to_type': multiclass_loaders['idx_to_type']
        }, f)
    
    print("\nTraining completed! Models saved in 'models/' directory")

if __name__ == "__main__":
    main()