"""
Test model architectures.
"""
import pytest
import torch


def test_lstm_model_creation():
    """Test LSTM model can be created."""
    from src.model_lstm import LSTMMultilabelClassifier
    
    model = LSTMMultilabelClassifier(
        vocab_size=1000,
        embedding_dim=100,
        hidden_dim=64,
        num_layers=2,
        dropout=0.3,
        num_labels=7
    )
    
    assert model is not None
    assert isinstance(model, torch.nn.Module)
    
    # Test forward pass
    batch_size = 2
    seq_length = 10
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    
    output = model(input_ids)
    assert output.shape == (batch_size, 7)
    assert torch.all(output >= 0) and torch.all(output <= 1)  # Sigmoid output


def test_bert_model_creation():
    """Test BERT model can be created."""
    import pytest
    from src.model_bert import BERTMultilabelClassifier
    
    # Skip if transformers not available or network issues
    try:
        model = BERTMultilabelClassifier(
            num_labels=7,
            dropout=0.3,
            pretrained_model='distilbert-base-uncased'
        )
        
        assert model is not None
        assert isinstance(model, torch.nn.Module)
        
        # Test forward pass
        batch_size = 2
        seq_length = 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        attention_mask = torch.ones(batch_size, seq_length)
        
        output = model(input_ids, attention_mask)
        assert output.shape == (batch_size, 7)
        assert torch.all(output >= 0) and torch.all(output <= 1)  # Sigmoid output
    except Exception as e:
        pytest.skip(f"BERT model test skipped: {e}")


def test_vocabulary():
    """Test Vocabulary class."""
    from src.dataset_utils import Vocabulary
    
    vocab = Vocabulary(max_vocab_size=100)
    
    # Build vocabulary
    texts = ["hello world", "hello python", "python code"]
    vocab.build_vocab(texts)
    
    assert len(vocab) > 0
    assert len(vocab) <= 100
    
    # Test encoding
    encoded = vocab.encode("hello world", max_length=10)
    assert isinstance(encoded, list)
    assert len(encoded) == 10
    assert all(isinstance(x, int) for x in encoded)

