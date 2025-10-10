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

---

## 🚀 Local Setup & Usage

This project uses the datasets in `data/raw/` and `data/processed/` and ships with Makefile shortcuts so you can run everything with a few commands. **Model artifacts are not committed** (they are git-ignored), so you'll train models locally.

### 1) Clone the repo
```bash
git clone https://github.com/OmondiKevin/toxicity-detector.git
cd toxicity-detector
```

### 2) Create & activate a Python virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3) Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4) Download NLTK data (required for text preprocessing)
```bash
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### 5) Verify the datasets
The repository includes the raw and processed datasets:
- `data/raw/hate_offensive_speech_detection.csv`
- `data/raw/sample_submission.csv`
- `data/processed/` (train/val/test splits - already included)

### 6) Train the LSTM model
> Model artifacts are not committed. This step will generate `models/lstm_multilabel.pth` locally.
```bash
make train-lstm
# Training time: ~45 minutes on CPU (Apple M1)
```

### 7) Train the BERT model
> This step will generate `models/bert_multilabel.pth` locally.
```bash
make train-bert
# Training time: ~4-5 hours on CPU (Apple M1)
```

### 8) Evaluate both models
```bash
make eval-multilabel
# Generates metrics, confusion matrices, and comparison charts
# Outputs saved to: evaluation_results/
```

### 9) Run the API (FastAPI / Uvicorn)
```bash
make api
# Server: http://127.0.0.1:8000
# Health check:   curl -s http://127.0.0.1:8000/health
# Sample request: curl -s -X POST http://127.0.0.1:8000/classify_multilabel_bert \
#   -H 'Content-Type: application/json' \
#   -d '{"text":"I hate you","threshold":0.5}'
```

### 10) Launch the Streamlit demo
```bash
make demo
# Opens a local UI to test the multilabel models interactively
```

---

## 🧰 Makefile Targets

```makefile
prepare-merged    # Merge raw datasets into multilabel format
split-multilabel  # Create train/val/test splits
train-lstm        # Train LSTM multilabel model (~45 min on M1 CPU)
train-bert        # Train BERT multilabel model (~4-5 hours on M1 CPU)
eval-multilabel   # Evaluate both models with comprehensive metrics
api               # Start FastAPI server (Uvicorn)
demo              # Launch Streamlit demo UI
```

## 🔒 Model Artifacts Policy

- Trained models (e.g., `models/lstm_multilabel.pth`, `models/bert_multilabel.pth`) are **not** stored in Git.
- Model files are git-ignored due to their large size (100MB+).
- To obtain models locally, run `make train-lstm` and/or `make train-bert`.
- Processed datasets (`data/processed/*.csv`) **ARE** included in the repository for convenience.

## ⚙️ Hardware Requirements

- **Minimum:** 8GB RAM, any CPU
- **Recommended:** 16GB+ RAM for faster training
- **GPU:** Optional (CUDA support will significantly speed up training)
- **Note:** This project was developed on Apple MacBook Pro M1 (8GB RAM) with CPU-only training

## ❓ Troubleshooting

- **Virtualenv not active?** Ensure you see `(.venv)` in your prompt. Re-activate: `source .venv/bin/activate`
- **Data not found?** The processed data is included in the repo. If missing, run: `make prepare-merged && make split-multilabel`
- **API can't find model?** Train first: `make train-lstm` or `make train-bert`
- **Port in use?** Stop other servers or run: `uvicorn src.api:app --reload --port 8001`
- **NLTK WordNet error?** Run: `python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"`
- **Out of memory during training?** Reduce batch size in `src/train_lstm.py` or `src/train_bert.py`

