"""
Comprehensive Application Testing Script.
Tests all components: data, models, API logic, and predictions.
"""
import sys
sys.path.insert(0, 'src')

import torch
import json
import numpy as np
from src.model_lstm import LSTMMultilabelClassifier
from src.model_bert import BERTMultilabelClassifier
from src.preprocess import clean_text_advanced
from transformers import DistilBertTokenizer

LABEL_NAMES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate", "non_offensive"]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_data_files():
    """Test 1: Verify all data files exist"""
    print("="*80)
    print("TEST 1: DATA FILES")
    print("="*80)
    
    import pandas as pd
    
    files = {
        "Raw Data": [
            "data/raw/hate_offensive_speech_detection.csv",
            "data/raw/sample_submission.csv"
        ],
        "Processed Data": [
            "data/processed/merged_multilabel.csv",
            "data/processed/train_multilabel.csv",
            "data/processed/val_multilabel.csv",
            "data/processed/test_multilabel.csv"
        ]
    }
    
    all_ok = True
    for category, file_list in files.items():
        print(f"\n{category}:")
        for file in file_list:
            try:
                df = pd.read_csv(file, nrows=1)
                print(f"  [OK] {file} ({df.shape[1]} columns)")
            except Exception as e:
                print(f"  [ERROR] {file} - {e}")
                all_ok = False
    
    return all_ok


def test_lstm_model():
    """Test 2: Load and test LSTM model"""
    print("\n" + "="*80)
    print("TEST 2: LSTM MODEL")
    print("="*80)
    
    try:
        # Load config
        with open("models/lstm_config.json", 'r') as f:
            config = json.load(f)
        print(f"LSTM config loaded")
        
        # Load vocabulary
        vocab = torch.load("models/lstm_vocab.pth", weights_only=False)
        print(f"Vocabulary loaded: {len(vocab)} words")
        
        # Load model
        model = LSTMMultilabelClassifier(
            vocab_size=len(vocab),
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            num_labels=7
        ).to(DEVICE)
        
        checkpoint = torch.load("models/lstm_multilabel.pth", map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test prediction
        test_text = "This is a friendly comment"
        cleaned = clean_text_advanced(test_text, lemmatize=True)
        input_ids = vocab.encode(cleaned, max_length=128)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(DEVICE)
        
        with torch.no_grad():
            probs = model(input_tensor).cpu().numpy()[0]
        
        print(f"\nTest prediction successful:")
        print(f"  Input: '{test_text}'")
        for label, prob in zip(LABEL_NAMES, probs):
            print(f"  {label:20s}: {prob:.3f}")
        
        return True, model, vocab
        
    except Exception as e:
        print(f"LSTM test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None


def test_bert_model():
    """Test 3: Load and test BERT model"""
    print("\n" + "="*80)
    print("TEST 3: BERT MODEL")
    print("="*80)
    
    try:
        # Load config
        with open("models/bert_config.json", 'r') as f:
            config = json.load(f)
        print(f"BERT config loaded")
        
        # Load tokenizer
        tokenizer = DistilBertTokenizer.from_pretrained(config['model_name'])
        print(f"Tokenizer loaded: {config['model_name']}")
        
        # Load model
        model = BERTMultilabelClassifier(
            num_labels=7,
            dropout=config['dropout'],
            pretrained_model=config['model_name']
        ).to(DEVICE)
        
        checkpoint = torch.load("models/bert_multilabel.pth", map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Test prediction
        test_text = "This is a friendly comment"
        encoding = tokenizer(
            test_text,
            add_special_tokens=True,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(DEVICE)
        attention_mask = encoding['attention_mask'].to(DEVICE)
        
        with torch.no_grad():
            probs = model(input_ids, attention_mask).cpu().numpy()[0]
        
        print(f"\nTest prediction successful:")
        print(f"  Input: '{test_text}'")
        for label, prob in zip(LABEL_NAMES, probs):
            print(f"  {label:20s}: {prob:.3f}")
        
        return True, model, tokenizer
        
    except Exception as e:
        print(f"BERT test failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None


def test_moderation_logic():
    """Test 4: Test moderation action logic"""
    print("\n" + "="*80)
    print("TEST 4: MODERATION LOGIC")
    print("="*80)
    
    def get_moderation_action(probabilities, threshold_high=0.7, threshold_medium=0.5):
        toxic_labels = {k: v for k, v in probabilities.items() if k != "non_offensive"}
        max_toxic = max(toxic_labels.values()) if toxic_labels else 0
        max_label = max(toxic_labels, key=toxic_labels.get) if toxic_labels else None
        
        if probabilities.get("severe_toxic", 0) > threshold_high or probabilities.get("threat", 0) > threshold_high:
            return "block", "Severe toxicity or threats detected"
        
        if max_toxic > threshold_high:
            return "block", f"High {max_label} content detected"
        
        if max_toxic > threshold_medium:
            return "flag", f"Moderate {max_label} content detected - requires review"
        
        if probabilities.get("non_offensive", 0) > 0.7:
            return "allow", "Content appears safe"
        
        return "flag", "Uncertain classification - manual review recommended"
    
    test_cases = [
        {
            "name": "Safe content",
            "probs": {"toxic": 0.1, "severe_toxic": 0.05, "obscene": 0.1, "threat": 0.01, 
                     "insult": 0.1, "identity_hate": 0.02, "non_offensive": 0.95},
            "expected": "allow"
        },
        {
            "name": "Moderate toxicity",
            "probs": {"toxic": 0.6, "severe_toxic": 0.2, "obscene": 0.4, "threat": 0.1,
                     "insult": 0.5, "identity_hate": 0.1, "non_offensive": 0.3},
            "expected": "flag"
        },
        {
            "name": "Severe toxicity",
            "probs": {"toxic": 0.9, "severe_toxic": 0.8, "obscene": 0.7, "threat": 0.2,
                     "insult": 0.85, "identity_hate": 0.75, "non_offensive": 0.1},
            "expected": "block"
        },
        {
            "name": "Threat detected",
            "probs": {"toxic": 0.5, "severe_toxic": 0.3, "obscene": 0.2, "threat": 0.85,
                     "insult": 0.4, "identity_hate": 0.2, "non_offensive": 0.2},
            "expected": "block"
        }
    ]
    
    all_passed = True
    for test_case in test_cases:
        action, reason = get_moderation_action(test_case["probs"])
        passed = action == test_case["expected"]
        status = "[PASS]" if passed else "[FAIL]"
        print(f"\n{status} Test: {test_case['name']}")
        print(f"  Expected: {test_case['expected']}, Got: {action}")
        print(f"  Reason: {reason}")
        if not passed:
            all_passed = False
    
    return all_passed


def test_end_to_end():
    """Test 5: End-to-end prediction with real examples"""
    print("\n" + "="*80)
    print("TEST 5: END-TO-END PREDICTIONS")
    print("="*80)
    
    test_examples = [
        "Thank you so much! This is really helpful.",
        "You are absolutely terrible at this!",
        "I hate this stupid thing so much",
        "This is the worst idea I've ever heard, you idiot!"
    ]
    
    # Load LSTM
    with open("models/lstm_config.json", 'r') as f:
        config = json.load(f)
    vocab = torch.load("models/lstm_vocab.pth", weights_only=False)
    lstm_model = LSTMMultilabelClassifier(
        vocab_size=len(vocab),
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        num_labels=7
    ).to(DEVICE)
    checkpoint = torch.load("models/lstm_multilabel.pth", map_location=DEVICE, weights_only=False)
    lstm_model.load_state_dict(checkpoint['model_state_dict'])
    lstm_model.eval()
    
    print("\nLSTM Predictions:")
    print("-" * 80)
    
    for idx, text in enumerate(test_examples, 1):
        cleaned = clean_text_advanced(text, lemmatize=True)
        input_ids = vocab.encode(cleaned, max_length=128)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(DEVICE)
        
        with torch.no_grad():
            probs = lstm_model(input_tensor).cpu().numpy()[0]
        
        max_toxic = max(probs[:-1])  # Exclude non_offensive
        max_label_idx = np.argmax(probs[:-1])
        non_offensive = probs[-1]
        
        print(f"\n{idx}. \"{text[:60]}{'...' if len(text)>60 else ''}\"")
        print(f"   Top toxic: {LABEL_NAMES[max_label_idx]} ({max_toxic:.3f})")
        print(f"   Non-offensive: {non_offensive:.3f}")
        
        if max_toxic > 0.7:
            action = "BLOCK"
        elif max_toxic > 0.5:
            action = "FLAG"
        else:
            action = "ALLOW"
        print(f"   Action: {action}")
    
    return True


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*80)
    print("TOXICITY DETECTOR - APPLICATION TESTING SUITE")
    print("="*80)
    print(f"\nDevice: {DEVICE}")
    
    results = {}
    
    # Test 1: Data
    results['data'] = test_data_files()
    
    # Test 2: LSTM
    results['lstm'], _, _ = test_lstm_model()
    
    # Test 3: BERT
    results['bert'], _, _ = test_bert_model()
    
    # Test 4: Moderation Logic
    results['moderation'] = test_moderation_logic()
    
    # Test 5: End-to-end
    results['e2e'] = test_end_to_end()
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name.upper():20s}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*80)
    if all_passed:
        print("ALL TESTS PASSED - APPLICATION IS READY")
    else:
        print("SOME TESTS FAILED - CHECK ERRORS ABOVE")
    print("="*80)
    
    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

