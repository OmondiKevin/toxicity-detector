"""
FastAPI service for multilabel toxicity classification.
Provides endpoints for LSTM and BERT model inference with content moderation.
"""
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Dict, Any, Optional
import torch
import json
import sys

APP_NAME = "toxicity-detector-api"

app = FastAPI(title=APP_NAME, version="1.0.0")

LSTM_MODEL = None
BERT_MODEL = None
LSTM_VOCAB = None
BERT_TOKENIZER = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LABEL_NAMES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate", "non_offensive"]


def load_lstm_model():
    """Lazy load LSTM model"""
    global LSTM_MODEL, LSTM_VOCAB
    if LSTM_MODEL is None:
        from src.model_lstm import LSTMMultilabelClassifier
        from src import dataset_utils

        # Compatibility layer: maps old module path for pickle compatibility
        sys.modules['dataset_utils'] = dataset_utils

        with open("models/lstm_config.json", 'r') as f:
            config = json.load(f)
        LSTM_VOCAB = torch.load("models/lstm_vocab.pth", weights_only=False)

        LSTM_MODEL = LSTMMultilabelClassifier(
            vocab_size=len(LSTM_VOCAB),
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            num_labels=7
        ).to(DEVICE)

        checkpoint = torch.load("models/lstm_multilabel.pth", map_location=DEVICE, weights_only=False)
        LSTM_MODEL.load_state_dict(checkpoint['model_state_dict'])
        LSTM_MODEL.eval()

    return LSTM_MODEL, LSTM_VOCAB


def load_bert_model():
    """Lazy load BERT model"""
    global BERT_MODEL, BERT_TOKENIZER
    if BERT_MODEL is None:
        from src.model_bert import BERTMultilabelClassifier
        from transformers import DistilBertTokenizer

        with open("models/bert_config.json", 'r') as f:
            config = json.load(f)

        BERT_TOKENIZER = DistilBertTokenizer.from_pretrained(config['model_name'])
        BERT_MODEL = BERTMultilabelClassifier(
            num_labels=7,
            dropout=config['dropout'],
            pretrained_model=config['model_name']
        ).to(DEVICE)

        checkpoint = torch.load("models/bert_multilabel.pth", map_location=DEVICE, weights_only=False)
        BERT_MODEL.load_state_dict(checkpoint['model_state_dict'])
        BERT_MODEL.eval()

    return BERT_MODEL, BERT_TOKENIZER


def get_moderation_action(probabilities: Dict[str, float], threshold_high=0.7, threshold_medium=0.5):
    """
    Determine moderation action based on toxicity probabilities.

    Returns:
        action: "block", "flag", or "allow"
        reason: explanation for the action
    """
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


class MLIn(BaseModel):
    text: str
    threshold: Optional[float] = 0.5


class MLOut(BaseModel):
    text: str
    probabilities: Dict[str, float]
    predictions: Dict[str, bool]
    action: str  # "allow", "flag", "block"
    reason: str
    model: str  # "lstm" or "bert"


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/version")
def version() -> Dict[str, Any]:
    return {"app": APP_NAME, "version": app.version}


@app.post("/classify_multilabel_lstm", response_model=MLOut)
def classify_multilabel_lstm(inp: MLIn) -> MLOut:
    """
    Multilabel classification using LSTM model.
    Returns probabilities for 7 toxicity categories and moderation action.
    """
    from src.preprocess import clean_text_advanced

    model, vocab = load_lstm_model()

    cleaned_text = clean_text_advanced(inp.text, lemmatize=True)

    input_ids = vocab.encode(cleaned_text, max_length=128)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        probs = model(input_tensor).cpu().numpy()[0]

    probabilities = {label: float(prob) for label, prob in zip(LABEL_NAMES, probs)}
    predictions = {label: bool(prob > inp.threshold) for label, prob in zip(LABEL_NAMES, probs)}

    action, reason = get_moderation_action(probabilities)

    return MLOut(
        text=inp.text,
        probabilities=probabilities,
        predictions=predictions,
        action=action,
        reason=reason,
        model="lstm"
    )


@app.post("/classify_multilabel_bert", response_model=MLOut)
def classify_multilabel_bert(inp: MLIn) -> MLOut:
    """
    Multilabel classification using BERT model.
    Returns probabilities for 7 toxicity categories and moderation action.
    """
    model, tokenizer = load_bert_model()

    encoding = tokenizer(
        inp.text,
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

    probabilities = {label: float(prob) for label, prob in zip(LABEL_NAMES, probs)}
    predictions = {label: bool(prob > inp.threshold) for label, prob in zip(LABEL_NAMES, probs)}

    action, reason = get_moderation_action(probabilities)

    return MLOut(
        text=inp.text,
        probabilities=probabilities,
        predictions=predictions,
        action=action,
        reason=reason,
        model="bert"
    )
