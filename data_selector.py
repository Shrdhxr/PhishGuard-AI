# data_selector.py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import pickle
import os

class EmailDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class DataSelector:
    def __init__(self):
        self.df_email = None
        self.df_type = None
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        
    def load_datasets(self):
        """Load both datasets from default locations"""
        try:
            # Load phishing email dataset
            self.df_email = pd.read_csv("data\Phishing_Email.csv")
            self.df_email = self.df_email.dropna(subset=['Email Text'])
            print(f"Loaded Phishing Email dataset: {len(self.df_email)} records")
            
            # Load phishing type dataset
            self.df_type = pd.read_csv("data\phishing_data_by_type.csv")
            self.df_type = self.df_type.dropna(subset=['Text'])
            print(f"Loaded Phishing Type dataset: {len(self.df_type)} records")
            
            return True
        except Exception as e:
            print(f"Error loading datasets: {e}")
            return False
    
    def preprocess_email_data(self):
        """Preprocess the phishing email dataset for binary classification"""
        df = self.df_email.copy()
        
        # Map labels to binary
        df['label'] = df['Email Type'].map({'Safe Email': 0, 'Phishing Email': 1})
        
        # Extract features and labels
        X = df['Email Text'].values
        y = df['label'].values
        
        print(f"Email dataset class distribution:")
        print(f"Safe Emails (0): {np.sum(y == 0)}")
        print(f"Phishing Emails (1): {np.sum(y == 1)}")
        
        return X, y
    
    def preprocess_type_data(self):
        """Preprocess the phishing type dataset for multi-class classification"""
        df = self.df_type.copy()
        
        # Extract features and labels
        X = df['Text'].values
        y = df['Type'].values
        
        # Create label mapping
        unique_types = np.unique(y)
        type_to_idx = {typ: idx for idx, typ in enumerate(unique_types)}
        idx_to_type = {idx: typ for typ, idx in type_to_idx.items()}
        
        y_encoded = np.array([type_to_idx[typ] for typ in y])
        
        print(f"Phishing types distribution:")
        for typ, idx in type_to_idx.items():
            count = np.sum(y_encoded == idx)
            print(f"{typ}: {count} samples")
        
        return X, y_encoded, type_to_idx, idx_to_type
    
    def prepare_data(self):
        """Prepare and save raw data without DataLoader objects"""
        # Process binary classification data
        X_email, y_email = self.preprocess_email_data()
        
        # Split binary data
        X_temp, X_test_email, y_temp, y_test_email = train_test_split(
            X_email, y_email, test_size=0.2, random_state=42, stratify=y_email
        )
        
        X_train_email, X_val_email, y_train_email, y_val_email = train_test_split(
            X_temp, y_temp, test_size=0.125, random_state=42, stratify=y_temp
        )
        
        # Process multi-class data
        X_type, y_type, type_to_idx, idx_to_type = self.preprocess_type_data()
        
        # Split multi-class data
        X_temp_type, X_test_type, y_temp_type, y_test_type = train_test_split(
            X_type, y_type, test_size=0.2, random_state=42, stratify=y_type
        )
        
        X_train_type, X_val_type, y_train_type, y_val_type = train_test_split(
            X_temp_type, y_temp_type, test_size=0.125, random_state=42, stratify=y_temp_type
        )
        
        # Prepare data dictionaries (save only raw data)
        binary_data = {
            'X_train': X_train_email,
            'X_val': X_val_email,
            'X_test': X_test_email,
            'y_train': y_train_email,
            'y_val': y_val_email,
            'y_test': y_test_email,
            'num_classes': 2
        }
        
        multiclass_data = {
            'X_train': X_train_type,
            'X_val': X_val_type,
            'X_test': X_test_type,
            'y_train': y_train_type,
            'y_val': y_val_type,
            'y_test': y_test_type,
            'num_classes': len(type_to_idx),
            'type_to_idx': type_to_idx,
            'idx_to_type': idx_to_type
        }
        
        # Create directories
        os.makedirs('data/processed', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        
        # Save data
        with open('data/processed/binary_data.pkl', 'wb') as f:
            pickle.dump(binary_data, f)
        
        with open('data/processed/multiclass_data.pkl', 'wb') as f:
            pickle.dump(multiclass_data, f)
        
        # Save tokenizer
        self.tokenizer.save_pretrained('models/tokenizer')
        
        print("Data processed successfully!")
        print(f"Binary - Train: {len(X_train_email)}, Val: {len(X_val_email)}, Test: {len(X_test_email)}")
        print(f"Multiclass - Train: {len(X_train_type)}, Val: {len(X_val_type)}, Test: {len(X_test_type)}")
        
        return binary_data, multiclass_data

def create_data_loaders(binary_data, multiclass_data, batch_size=16):
    """Create DataLoader objects from raw data"""
    tokenizer = AutoTokenizer.from_pretrained('models/tokenizer')
    
    # Create datasets for binary classification
    train_dataset_binary = EmailDataset(
        binary_data['X_train'], binary_data['y_train'], tokenizer
    )
    val_dataset_binary = EmailDataset(
        binary_data['X_val'], binary_data['y_val'], tokenizer
    )
    test_dataset_binary = EmailDataset(
        binary_data['X_test'], binary_data['y_test'], tokenizer
    )
    
    # Create datasets for multiclass classification
    train_dataset_multiclass = EmailDataset(
        multiclass_data['X_train'], multiclass_data['y_train'], tokenizer
    )
    val_dataset_multiclass = EmailDataset(
        multiclass_data['X_val'], multiclass_data['y_val'], tokenizer
    )
    test_dataset_multiclass = EmailDataset(
        multiclass_data['X_test'], multiclass_data['y_test'], tokenizer
    )
    
    # Create data loaders
    binary_loaders = {
        'train_loader': DataLoader(train_dataset_binary, batch_size=batch_size, shuffle=True),
        'val_loader': DataLoader(val_dataset_binary, batch_size=batch_size, shuffle=False),
        'test_loader': DataLoader(test_dataset_binary, batch_size=batch_size, shuffle=False),
        'num_classes': binary_data['num_classes']
    }
    
    multiclass_loaders = {
        'train_loader': DataLoader(train_dataset_multiclass, batch_size=batch_size, shuffle=True),
        'val_loader': DataLoader(val_dataset_multiclass, batch_size=batch_size, shuffle=False),
        'test_loader': DataLoader(test_dataset_multiclass, batch_size=batch_size, shuffle=False),
        'num_classes': multiclass_data['num_classes'],
        'type_to_idx': multiclass_data['type_to_idx'],
        'idx_to_type': multiclass_data['idx_to_type']
    }
    
    return binary_loaders, multiclass_loaders

if __name__ == "__main__":
    selector = DataSelector()
    if selector.load_datasets():
        binary_data, multiclass_data = selector.prepare_data()
        print("Data preparation completed!")
    else:
        print("Failed to load datasets")