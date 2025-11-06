"""
LSTM-based multi-label text classifier.
Uses trainable embeddings (can be initialized with GloVe/Word2Vec).
"""
import torch
import torch.nn as nn


class LSTMMultilabelClassifier(nn.Module):
    """
    BiLSTM with attention for multi-label text classification.

    Architecture:
    - Embedding layer (trainable)
    - Bidirectional LSTM
    - Attention mechanism
    - Fully connected layers
    - Sigmoid activation for multi-label output
    """

    def __init__(
        self,
        vocab_size,
        embedding_dim=200,
        hidden_dim=128,
        num_labels=7,
        num_layers=2,
        dropout=0.3,
        bidirectional=True
    ):
        super(LSTMMultilabelClassifier, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.attention = nn.Linear(hidden_dim * self.num_directions, 1)
        self.dropout = nn.Dropout(dropout)

        fc_input_dim = hidden_dim * self.num_directions
        self.fc1 = nn.Linear(fc_input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc_out = nn.Linear(64, num_labels)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def attention_layer(self, lstm_output):
        """
        Apply attention mechanism to LSTM outputs.

        Args:
            lstm_output: (batch_size, seq_len, hidden_dim * num_directions)

        Returns:
            context_vector: (batch_size, hidden_dim * num_directions)
        """
        attention_weights = torch.softmax(
            self.attention(lstm_output).squeeze(-1),
            dim=1
        )

        context_vector = torch.sum(
            attention_weights.unsqueeze(-1) * lstm_output,
            dim=1
        )

        return context_vector

    def forward(self, input_ids):
        """
        Forward pass.

        Args:
            input_ids: (batch_size, seq_len)

        Returns:
            logits: (batch_size, num_labels) - sigmoid probabilities
        """
        embedded = self.embedding(input_ids)
        embedded = self.dropout(embedded)

        lstm_out, (hidden, cell) = self.lstm(embedded)

        context = self.attention_layer(lstm_out)
        context = self.dropout(context)

        x = self.relu(self.fc1(context))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        logits = self.fc_out(x)

        probs = self.sigmoid(logits)

        return probs

    def get_embedding_weights(self):
        """Get embedding layer weights for initialization."""
        return self.embedding.weight.data

    def set_embedding_weights(self, weights):
        """Set embedding layer weights (e.g., from pre-trained GloVe)."""
        self.embedding.weight.data.copy_(torch.from_numpy(weights))
