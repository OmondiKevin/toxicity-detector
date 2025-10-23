"""
BERT-based multi-label text classifier.
Fine-tunes DistilBERT for efficiency while maintaining performance.
"""
import torch
import torch.nn as nn
from transformers import DistilBertModel

class BERTMultilabelClassifier(nn.Module):
    """
    DistilBERT-based multi-label classifier.
    
    Architecture:
    - DistilBERT encoder (fine-tuned)
    - Dropout
    - Fully connected classification head
    - Sigmoid activation for multi-label output
    """
    
    def __init__(
        self,
        num_labels=7,
        dropout=0.3,
        pretrained_model='distilbert-base-uncased'
    ):
        super(BERTMultilabelClassifier, self).__init__()
        
        # Load pre-trained DistilBERT
        self.bert = DistilBertModel.from_pretrained(pretrained_model)
        
        # Get hidden size from BERT config
        self.hidden_size = self.bert.config.hidden_size  # 768 for DistilBERT
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Classification head
        self.fc1 = nn.Linear(self.hidden_size, 256)
        self.fc2 = nn.Linear(256, num_labels)
        
        # Activations
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_ids, attention_mask):
        """
        Forward pass.
        
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            
        Returns:
            probs: (batch_size, num_labels) - sigmoid probabilities
        """
        # BERT encoding
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]  # (batch_size, hidden_size)
        
        # Dropout
        pooled_output = self.dropout(pooled_output)
        
        # Classification head
        x = self.relu(self.fc1(pooled_output))
        x = self.dropout(x)
        logits = self.fc2(x)
        
        # Sigmoid for multi-label
        probs = self.sigmoid(logits)
        
        return probs
    
    def freeze_bert_encoder(self):
        """Freeze BERT encoder weights (only train classification head)."""
        for param in self.bert.parameters():
            param.requires_grad = False
    
    def unfreeze_bert_encoder(self):
        """Unfreeze BERT encoder for fine-tuning."""
        for param in self.bert.parameters():
            param.requires_grad = True

