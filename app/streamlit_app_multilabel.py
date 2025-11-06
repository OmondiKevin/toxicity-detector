"""
Enhanced Streamlit Demo with Multilabel Classification.
Supports LSTM and BERT models with content moderation actions.
"""
import streamlit as st
import torch
import json
import sys
sys.path.insert(0, 'src')

from model_lstm import LSTMMultilabelClassifier
from model_bert import BERTMultilabelClassifier
from preprocess import clean_text_advanced
from transformers import DistilBertTokenizer

st.set_page_config(
    page_title="Toxicity Detector Pro",
    page_icon=":shield:",
    layout="wide"
)

LABEL_NAMES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate", "non_offensive"]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@st.cache_resource
def load_lstm_multilabel():
    """Load LSTM multilabel model"""
    with open("models/lstm_config.json", 'r') as f:
        config = json.load(f)
    
    vocab = torch.load("models/lstm_vocab.pth", weights_only=False)
    
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
    
    return model, vocab


@st.cache_resource
def load_bert_multilabel():
    """Load BERT multilabel model"""
    with open("models/bert_config.json", 'r') as f:
        config = json.load(f)
    
    tokenizer = DistilBertTokenizer.from_pretrained(config['model_name'])
    
    model = BERTMultilabelClassifier(
        num_labels=7,
        dropout=config['dropout'],
        pretrained_model=config['model_name']
    ).to(DEVICE)
    
    checkpoint = torch.load("models/bert_multilabel.pth", map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, tokenizer


def get_moderation_action(probs_dict, threshold_high=0.7, threshold_medium=0.5):
    """Determine moderation action"""
    toxic_labels = {k: v for k, v in probs_dict.items() if k != "non_offensive"}
    max_toxic = max(toxic_labels.values()) if toxic_labels else 0
    max_label = max(toxic_labels, key=toxic_labels.get) if toxic_labels else None
    
    if probs_dict.get("severe_toxic", 0) > threshold_high or probs_dict.get("threat", 0) > threshold_high:
        return "BLOCK", "Severe toxicity or threats detected", "danger"
    
    if max_toxic > threshold_high:
        return "BLOCK", f"High {max_label} content detected", "danger"
    
    if max_toxic > threshold_medium:
        return "FLAG", f"Moderate {max_label} - requires review", "warning"
    
    if probs_dict.get("non_offensive", 0) > 0.7:
        return "ALLOW", "Content appears safe", "success"
    
    return "FLAG", "Uncertain - manual review recommended", "warning"


def predict_lstm(text, model, vocab):
    """Predict with LSTM model"""
    cleaned = clean_text_advanced(text, lemmatize=True)
    input_ids = vocab.encode(cleaned, max_length=128)
    input_tensor = torch.tensor([input_ids], dtype=torch.long).to(DEVICE)
    
    with torch.no_grad():
        probs = model(input_tensor).cpu().numpy()[0]
    
    return {label: float(prob) for label, prob in zip(LABEL_NAMES, probs)}


def predict_bert(text, model, tokenizer):
    """Predict with BERT model"""
    encoding = tokenizer(
        text,
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
    
    return {label: float(prob) for label, prob in zip(LABEL_NAMES, probs)}


# ====== UI ======
st.title("Toxicity Detector Pro")
st.markdown("**Multilabel content moderation** using deep learning (LSTM & BERT)")

# Tabs
tab1, tab2 = st.tabs(["LSTM Model", "BERT Model"])

# ====== TAB 1: LSTM ======
with tab1:
    st.header("BiLSTM with Attention")
    st.caption("Fast inference with custom embeddings")
    
    try:
        lstm_model, lstm_vocab = load_lstm_multilabel()
        
        text_lstm = st.text_area(
            "Enter text to analyze:",
            height=120,
            key="lstm_input",
            placeholder="Type or paste content here..."
        )
        
        threshold_lstm = st.slider("Detection threshold:", 0.0, 1.0, 0.5, 0.05, key="lstm_threshold")
        
        if st.button("Analyze with LSTM", key="lstm_btn"):
            if text_lstm.strip():
                with st.spinner("Analyzing..."):
                    probs = predict_lstm(text_lstm, lstm_model, lstm_vocab)
                    action, reason, action_type = get_moderation_action(probs)
                    
                    # Show action
                    if action_type == "danger":
                        st.error(f"**{action}**: {reason}")
                    elif action_type == "warning":
                        st.warning(f"**{action}**: {reason}")
                    else:
                        st.success(f"**{action}**: {reason}")
                    
                    # Show probabilities
                    st.subheader("Toxicity Probabilities")
                    cols = st.columns(4)
                    for idx, (label, prob) in enumerate(probs.items()):
                        col = cols[idx % 4]
                        is_detected = prob > threshold_lstm
                        col.metric(
                            label.replace("_", " ").title(),
                            f"{prob:.1%}",
                            delta=("detected" if is_detected else ""),
                            delta_color="inverse" if label != "non_offensive" else "normal"
                        )
                    
                    # Progress bars
                    st.subheader("Detailed Breakdown")
                    for label, prob in probs.items():
                        st.write(f"**{label.replace('_', ' ').title()}**")
                        st.progress(prob)
            else:
                st.info("Please enter some text to analyze")
    
    except Exception as e:
        st.error(f"Could not load LSTM model: {e}")

# ====== TAB 2: BERT ======
with tab2:
    st.header("DistilBERT Fine-tuned")
    st.caption("State-of-the-art transformer model")
    
    try:
        bert_model, bert_tokenizer = load_bert_multilabel()
        
        text_bert = st.text_area(
            "Enter text to analyze:",
            height=120,
            key="bert_input",
            placeholder="Type or paste content here..."
        )
        
        threshold_bert = st.slider("Detection threshold:", 0.0, 1.0, 0.5, 0.05, key="bert_threshold")
        
        if st.button("Analyze with BERT", key="bert_btn"):
            if text_bert.strip():
                with st.spinner("Analyzing..."):
                    probs = predict_bert(text_bert, bert_model, bert_tokenizer)
                    action, reason, action_type = get_moderation_action(probs)
                    
                    # Show action
                    if action_type == "danger":
                        st.error(f"**{action}**: {reason}")
                    elif action_type == "warning":
                        st.warning(f"**{action}**: {reason}")
                    else:
                        st.success(f"**{action}**: {reason}")
                    
                    # Show probabilities
                    st.subheader("Toxicity Probabilities")
                    cols = st.columns(4)
                    for idx, (label, prob) in enumerate(probs.items()):
                        col = cols[idx % 4]
                        is_detected = prob > threshold_bert
                        col.metric(
                            label.replace("_", " ").title(),
                            f"{prob:.1%}",
                            delta=("detected" if is_detected else ""),
                            delta_color="inverse" if label != "non_offensive" else "normal"
                        )
                    
                    # Progress bars
                    st.subheader("Detailed Breakdown")
                    for label, prob in probs.items():
                        st.write(f"**{label.replace('_', ' ').title()}**")
                        st.progress(prob)
            else:
                st.info("Please enter some text to analyze")
    
    except Exception as e:
        st.error(f"Could not load BERT model: {e}")

# ====== Footer ======
st.divider()
st.caption("Built for the Toxicity Detection Interview Task | Models: LSTM (4.8M params) and BERT (66M+ params)")

