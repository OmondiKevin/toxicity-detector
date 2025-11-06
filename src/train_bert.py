"""
Train BERT multi-label classifier.
Uses DistilBERT for efficiency.
"""
import os
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm

from model_bert import BERTMultilabelClassifier
from dataset_utils import create_bert_dataloaders

# Paths
TRAIN_PATH = "data/processed/train_multilabel.csv"
VAL_PATH = "data/processed/val_multilabel.csv"
TEST_PATH = "data/processed/test_multilabel.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "bert_multilabel.pth")
CONFIG_PATH = os.path.join(MODEL_DIR, "bert_config.json")

# Hyperparameters
HYPERPARAMS = {
    "model_name": "distilbert-base-uncased",
    "dropout": 0.3,
    "max_length": 128,
    "batch_size": 16,  # Smaller for BERT
    "learning_rate": 2e-5,  # Smaller LR for fine-tuning
    "num_epochs": 10,
    "patience": 3,
    "warmup_steps": 100
}

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            # Store predictions and labels
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    return avg_loss, all_preds, all_labels


def run_training():
    """Main training function."""
    print("="*80)
    print("BERT TRAINING")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print("\nLoading data...")
    train_loader, val_loader, test_loader, _ = create_bert_dataloaders(
        TRAIN_PATH,
        VAL_PATH,
        TEST_PATH,
        model_name=HYPERPARAMS["model_name"],
        max_length=HYPERPARAMS["max_length"],
        batch_size=HYPERPARAMS["batch_size"]
    )
    
    print(f"Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")
    
    print("\nCreating BERT model...")
    model = BERTMultilabelClassifier(
        num_labels=7,
        dropout=HYPERPARAMS["dropout"],
        pretrained_model=HYPERPARAMS["model_name"]
    ).to(device)
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = nn.BCELoss()
    
    # Different learning rates for encoder vs head
    optimizer = optim.AdamW([
        {'params': model.bert.parameters(), 'lr': HYPERPARAMS["learning_rate"]},
        {'params': model.fc1.parameters(), 'lr': HYPERPARAMS["learning_rate"] * 10},
        {'params': model.fc2.parameters(), 'lr': HYPERPARAMS["learning_rate"] * 10}
    ])
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    print(f"\nTraining for {HYPERPARAMS['num_epochs']} epochs...")
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(HYPERPARAMS["num_epochs"]):
        print(f"\nEpoch {epoch + 1}/{HYPERPARAMS['num_epochs']}")
        print("-" * 80)
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, _, _ = evaluate(model, val_loader, criterion, device)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            print(f"New best model! Saving...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'hyperparams': HYPERPARAMS
            }, MODEL_PATH)
            
            # Save config
            with open(CONFIG_PATH, 'w') as f:
                json.dump(HYPERPARAMS, f, indent=2)
            
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{HYPERPARAMS['patience']}")
            
            if patience_counter >= HYPERPARAMS["patience"]:
                print("\nEarly stopping triggered")
                break
    
    print("\nLoading best model for final evaluation...")
    checkpoint = torch.load(MODEL_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("Evaluating on test set...")
    test_loss, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")
    
    # Save test predictions
    np.save(os.path.join(MODEL_DIR, "bert_test_preds.npy"), test_preds)
    np.save(os.path.join(MODEL_DIR, "bert_test_labels.npy"), test_labels)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED")
    print(f"Best val loss: {best_val_loss:.4f}, Test loss: {test_loss:.4f}")
    print(f"Saved to: {MODEL_PATH}")
    print("="*80)
    
    return model, test_preds, test_labels


if __name__ == "__main__":
    run_training()

