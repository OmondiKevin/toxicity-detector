"""
Dataset and DataLoader utilities for multi-label classification.
Supports both LSTM (tokenization) and BERT (transformer tokenization).
"""
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from transformers import DistilBertTokenizer
from collections import Counter

LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate", "non_offensive"]


class Vocabulary:
    """Vocabulary for LSTM model."""

    def __init__(self, max_vocab_size=20000):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.max_vocab_size = max_vocab_size

    def build_vocab(self, texts):
        """Build vocabulary from texts."""
        word_counts = Counter()
        for text in texts:
            # Handle potential non-string values
            if isinstance(text, str):
                words = text.split()
                word_counts.update(words)

        # Keep most common words
        most_common = word_counts.most_common(self.max_vocab_size - 2)  # -2 for PAD and UNK

        for idx, (word, count) in enumerate(most_common, start=2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word

        print(f"Vocabulary built: {len(self.word2idx)} words")
        return self

    def encode(self, text, max_length=128):
        """Encode text to indices."""
        # Handle potential non-string values
        if not isinstance(text, str):
            text = str(text) if text is not None else ""

        words = text.split()[:max_length]
        indices = [self.word2idx.get(word, 1) for word in words]  # 1 is <UNK>

        # Pad to max_length
        if len(indices) < max_length:
            indices += [0] * (max_length - len(indices))  # 0 is <PAD>

        return indices

    def __len__(self):
        return len(self.word2idx)


class LSTMDataset(Dataset):
    """Dataset for LSTM model."""

    def __init__(self, texts, labels, vocab, max_length=128):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Encode text
        input_ids = self.vocab.encode(text, self.max_length)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.float32)
        }


class BERTDataset(Dataset):
    """Dataset for BERT model."""

    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Handle potential non-string values
        if not isinstance(text, str):
            text = str(text) if text is not None else ""

        # Tokenize with BERT tokenizer
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.float32)
        }


def load_data(csv_path, label_cols=LABEL_COLS):
    """
    Load data from CSV.

    Returns:
        texts: list of strings
        labels: numpy array of shape (n_samples, n_labels)
    """
    df = pd.read_csv(csv_path)
    texts = df['text'].tolist()
    labels = df[label_cols].values.astype(np.float32)
    return texts, labels


def create_lstm_dataloaders(
    train_path,
    val_path,
    test_path,
    vocab=None,
    max_vocab_size=20000,
    max_length=128,
    batch_size=32
):
    """
    Create DataLoaders for LSTM model.

    Args:
        train_path: Path to training CSV
        val_path: Path to validation CSV
        test_path: Path to test CSV
        vocab: Existing vocabulary (if None, will build from train)
        max_vocab_size: Maximum vocabulary size
        max_length: Maximum sequence length
        batch_size: Batch size

    Returns:
        train_loader, val_loader, test_loader, vocab
    """
    # Load data
    train_texts, train_labels = load_data(train_path)
    val_texts, val_labels = load_data(val_path)
    test_texts, test_labels = load_data(test_path)

    # Build vocabulary from training data
    if vocab is None:
        vocab = Vocabulary(max_vocab_size=max_vocab_size)
        vocab.build_vocab(train_texts)

    # Create datasets
    train_dataset = LSTMDataset(train_texts, train_labels, vocab, max_length)
    val_dataset = LSTMDataset(val_texts, val_labels, vocab, max_length)
    test_dataset = LSTMDataset(test_texts, test_labels, vocab, max_length)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, vocab


def create_bert_dataloaders(
    train_path,
    val_path,
    test_path,
    model_name='distilbert-base-uncased',
    max_length=128,
    batch_size=16
):
    """
    Create DataLoaders for BERT model.

    Args:
        train_path: Path to training CSV
        val_path: Path to validation CSV
        test_path: Path to test CSV
        model_name: BERT model name
        max_length: Maximum sequence length
        batch_size: Batch size (smaller for BERT due to memory)

    Returns:
        train_loader, val_loader, test_loader, tokenizer
    """
    # Load tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    # Load data
    train_texts, train_labels = load_data(train_path)
    val_texts, val_labels = load_data(val_path)
    test_texts, test_labels = load_data(test_path)

    # Create datasets
    train_dataset = BERTDataset(train_texts, train_labels, tokenizer, max_length)
    val_dataset = BERTDataset(val_texts, val_labels, tokenizer, max_length)
    test_dataset = BERTDataset(test_texts, test_labels, tokenizer, max_length)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, tokenizer
