"""
Test that all modules can be imported correctly.
"""
import pytest


def test_import_src_modules():
    """Test that all src modules can be imported."""
    from src import model_lstm
    from src import model_bert
    from src import dataset_utils
    from src import preprocess
    from src import api
    
    assert model_lstm is not None
    assert model_bert is not None
    assert dataset_utils is not None
    assert preprocess is not None
    assert api is not None


def test_import_classes():
    """Test that main classes can be imported."""
    from src.model_lstm import LSTMMultilabelClassifier
    from src.model_bert import BERTMultilabelClassifier
    from src.dataset_utils import Vocabulary, LSTMDataset, BERTDataset
    from src.preprocess import clean_text, clean_text_advanced
    
    assert LSTMMultilabelClassifier is not None
    assert BERTMultilabelClassifier is not None
    assert Vocabulary is not None
    assert LSTMDataset is not None
    assert BERTDataset is not None
    assert clean_text is not None
    assert clean_text_advanced is not None


def test_preprocess_functions():
    """Test preprocessing functions work."""
    from src.preprocess import clean_text, clean_text_advanced
    
    # Test basic cleaning
    test_text = "This is a test! @user #hashtag https://example.com"
    cleaned = clean_text(test_text)
    assert isinstance(cleaned, str)
    assert "@user" not in cleaned
    assert "#hashtag" not in cleaned
    assert "https://example.com" not in cleaned
    
    # Test advanced cleaning
    cleaned_advanced = clean_text_advanced(test_text, lemmatize=False)
    assert isinstance(cleaned_advanced, str)
    
    cleaned_advanced_lemma = clean_text_advanced(test_text, lemmatize=True)
    assert isinstance(cleaned_advanced_lemma, str)

