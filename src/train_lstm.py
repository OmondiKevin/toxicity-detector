"""
Train LSTM multi-label classifier.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import os
import json

from model_lstm import LSTMMultilabelClassifier
from dataset_utils import create_lstm_dataloaders

# Paths
TRAIN_PATH = "data/processed/train_multilabel.csv"
VAL_PATH = "data/processed/val_multilabel.csv"
TEST_PATH = "data/processed/test_multilabel.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "lstm_multilabel.pth")
VOCAB_PATH = os.path.join(MODEL_DIR, "lstm_vocab.pth")
CONFIG_PATH = os.path.join(MODEL_DIR, "lstm_config.json")

# Hyperparameters
HYPERPARAMS = {
    "max_vocab_size": 20000,
    "embedding_dim": 200,
    "hidden_dim": 128,
    "num_layers": 2,
    "dropout": 0.3,
    "max_length": 128,
    "batch_size": 64,
    "learning_rate": 0.001,
    "num_epochs": 15,
    "patience": 3
}

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        outputs = model(input_ids)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
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
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(input_ids)
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
    print("LSTM TRAINING")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print("\nLoading data...")
    train_loader, val_loader, test_loader, vocab = create_lstm_dataloaders(
        TRAIN_PATH,
        VAL_PATH,
        TEST_PATH,
        max_vocab_size=HYPERPARAMS["max_vocab_size"],
        max_length=HYPERPARAMS["max_length"],
        batch_size=HYPERPARAMS["batch_size"]
    )
    
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")
    
    print("\nCreating LSTM model...")
    model = LSTMMultilabelClassifier(
        vocab_size=len(vocab),
        embedding_dim=HYPERPARAMS["embedding_dim"],
        hidden_dim=HYPERPARAMS["hidden_dim"],
        num_layers=HYPERPARAMS["num_layers"],
        dropout=HYPERPARAMS["dropout"],
        num_labels=7
    ).to(device)
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=HYPERPARAMS["learning_rate"])
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
        val_loss, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
        
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
            
            # Save vocabulary
            torch.save(vocab, VOCAB_PATH)
            
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
    np.save(os.path.join(MODEL_DIR, "lstm_test_preds.npy"), test_preds)
    np.save(os.path.join(MODEL_DIR, "lstm_test_labels.npy"), test_labels)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETED")
    print(f"Best val loss: {best_val_loss:.4f}, Test loss: {test_loss:.4f}")
    print(f"Saved to: {MODEL_PATH}")
    print("="*80)
    
    return model, test_preds, test_labels


if __name__ == "__main__":
    run_training()

