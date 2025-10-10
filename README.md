# Toxicity Detector

A complete toxicity detection system with multi-label deep learning models, comprehensive evaluation, and production-ready API deployment.

## Interview Tasks: 5/5 Completed

| Task | Status | Key Deliverables |
|------|--------|------------------|
| **A. Data Preprocessing** | COMPLETED | 20,996 samples, 7 labels, train/val/test splits |
| **B. Model Building** | COMPLETED | LSTM (4.8M params) & BERT (66M+ params) |
| **C. Model Evaluation** | COMPLETED | Metrics, confusion matrices, ROC curves |
| **D. Deployment** | COMPLETED | FastAPI + Streamlit with moderation logic |
| **E. Documentation** | COMPLETED | Technical report + comprehensive README |

**Performance Summary:**
- **BERT ROC-AUC:** 0.85 (best discriminative power)
- **LSTM F1-Score:** 0.65 (best balance)
- **7-label multi-label classification** with automated moderation
- **Production-ready API** with real-time inference

---

## Overview

A complete multi-label toxicity detection system using deep learning (LSTM & BERT) with comprehensive evaluation and production-ready API deployment.

## Data Sources

This project uses only the files provided under `data/raw/`:
- `hate_offensive_speech_detection.csv` - Text source with 21,009 samples
- `sample_submission.csv` - Multi-label schema (7 toxicity categories)

No external downloads are required. Intelligent label mapping creates training data for all 7 toxicity categories.

## Quick Start

### 1. Data Preparation
```bash
make prepare-merged  # Merge datasets and create multilabel structure
```

### 2. Train Models
```bash
make train-lstm  # Train LSTM model (~15 min on CPU)
make train-bert  # Train BERT model (~1-2 hours on CPU)
```

### 3. Evaluate
```bash
make eval-multilabel  # Generate metrics and visualizations
```

### 4. Deploy
```bash
make api   # Start FastAPI server (port 8000)
make demo  # Launch Streamlit demo
```

## Available Targets

- `prepare-merged` - Prepare merged multilabel dataset
- `split-multilabel` - Create train/val/test splits
- `train-lstm` - Train LSTM multilabel model
- `train-bert` - Train BERT multilabel model
- `eval-multilabel` - Evaluate multilabel models with metrics & visualizations
- `api` - Start FastAPI server (port 8000)
- `demo` - Launch Streamlit demo application

## Project Structure

```
├── app/
│   └── streamlit_app.py       # Streamlit UI
├── data/
│   ├── processed/              # Cleaned train/val/test splits
│   └── raw/                    # Original datasets
│       ├── hate_offensive_speech_detection.csv
│       └── sample_submission.csv
├── models/                     # Trained models
│   ├── lstm_multilabel.pth
│   ├── bert_multilabel.pth
│   └── *.json (configs)
├── src/
│   ├── api.py                  # FastAPI service
│   ├── dataset_utils.py        # PyTorch datasets & dataloaders
│   ├── evaluate_multilabel.py  # Model evaluation & metrics
│   ├── model_bert.py           # BERT architecture
│   ├── model_lstm.py           # LSTM architecture
│   ├── prepare_merged.py       # Merge datasets
│   ├── preprocess.py           # Text cleaning utilities
│   ├── split_multilabel.py     # Train/val/test split
│   ├── train_bert.py           # BERT training
│   └── train_lstm.py           # LSTM training
└── requirements.txt
```

## API Endpoints

- `GET /health` - Health check
- `GET /version` - Version info
- `POST /classify_multilabel_lstm` - Multilabel classification with LSTM model
- `POST /classify_multilabel_bert` - Multilabel classification with BERT model

## Interview Task Implementation

### Task A: Data Preprocessing [COMPLETED]

**Objective:** Merge datasets and prepare for multi-label classification

**Implementation:**
- **Datasets merged:** Combined `hate_offensive_speech_detection.csv` (text source) with `sample_submission.csv` (label schema)
- **Text cleaning:** Removed URLs, mentions, hashtags, emojis, punctuation
- **Normalization:** Lowercasing + lemmatization (NLTK)
- **Label mapping strategy:**
  - Label 1 (hate) → `toxic`, `severe_toxic`, `insult`, `identity_hate`
  - Label 2 (offensive) → `toxic`, `obscene`, `insult`
  - Label 3 (neutral) → `non_offensive` only

**Output:** `data/processed/merged_multilabel.csv`
- 20,996 samples with 7 multi-label categories
- Ready for deep learning model training

**Usage:**
```bash
make prepare-merged
```

**Note:** Since `sample_submission.csv` is a template without text data, we used intelligent heuristics to map the 3-class labels from hate_offensive dataset to the required 7-class multi-label structure.

### Task B: Model Building [COMPLETED]

**Objective:** Build multi-label deep learning classifiers

**Models Implemented:**
1. **LSTM (BiLSTM with Attention)**
   - Parameters: 4.8M
   - Best Val Loss: 0.3622
   - Test Loss: 0.3624
   
2. **BERT (DistilBERT Fine-tuned)**
   - Parameters: 66M+
   - Best Val Loss: 0.3716
   - Test Loss: 0.3686

**Usage:**
```bash
make train-lstm  # Train LSTM model
make train-bert  # Train BERT model
```

### Task C: Model Evaluation [COMPLETED]

**Objective:** Compare model performance with comprehensive metrics

**Results Summary:**

| Metric | LSTM | BERT | Winner |
|--------|------|------|--------|
| Macro F1-Score | 0.3764 | 0.3592 | LSTM |
| Macro Precision | 0.5245 | 0.6221 | BERT |
| Macro Recall | 0.3671 | 0.3128 | LSTM |
| Macro ROC-AUC | 0.8054 | 0.8492 | BERT |
| Micro F1-Score | 0.6459 | 0.6385 | LSTM |

**Per-Label Performance (F1-Score):**
- `toxic`: LSTM 0.67 | BERT 0.59
- `severe_toxic`: LSTM 0.06 | BERT 0.09
- `obscene`: LSTM 0.42 | BERT 0.31
- `insult`: LSTM 0.67 | BERT 0.61
- `identity_hate`: LSTM 0.02 | BERT 0.08
- `non_offensive`: LSTM 0.80 | BERT 0.84

**Key Findings:**
- BERT achieves better ROC-AUC (0.85 vs 0.81)
- LSTM shows better overall F1 and recall
- Both models struggle with rare classes (`severe_toxic`, `identity_hate`, `threat`)
- `non_offensive` class performs best for both models

**Usage:**
```bash
make eval-multilabel  # Run comprehensive evaluation
```

**Deliverables:**
- `evaluation_results/evaluation_report.txt` - Detailed metrics
- `evaluation_results/metrics.json` - Machine-readable metrics
- `evaluation_results/lstm_confusion_matrices.png` - Per-label confusion matrices
- `evaluation_results/bert_confusion_matrices.png` - Per-label confusion matrices
- `evaluation_results/model_comparison.png` - Side-by-side comparison charts

### Task D: Deployment [COMPLETED]

**Objective:** Design and implement API for real-time content moderation

**API Endpoints:**

1. **`POST /classify_multilabel_lstm`** - LSTM-based classification
2. **`POST /classify_multilabel_bert`** - BERT-based classification

**Request Format:**
```json
{
  "text": "Your text content here",
  "threshold": 0.5
}
```

**Response Format:**
```json
{
  "text": "Your text content here",
  "probabilities": {
    "toxic": 0.23,
    "severe_toxic": 0.05,
    "obscene": 0.12,
    "threat": 0.01,
    "insult": 0.18,
    "identity_hate": 0.03,
    "non_offensive": 0.82
  },
  "predictions": {
    "toxic": false,
    "severe_toxic": false,
    ...
  },
  "action": "allow",
  "reason": "Content appears safe",
  "model": "lstm"
}
```

**Moderation Actions:**
- **BLOCK**: Severe toxicity detected (>70% confidence)
- **FLAG**: Moderate toxicity detected (50-70% confidence) - requires manual review
- **ALLOW**: Content appears safe

**Usage:**
```bash
make api   # Start FastAPI server (port 8000)
make demo  # Launch Streamlit demo
```

**Demo Features:**
- Real-time toxicity analysis
- Visual probability breakdown per category
- Automated moderation recommendations
- Side-by-side model comparison (LSTM vs BERT)
- Interactive threshold adjustment

## License

See LICENSE file for details.

