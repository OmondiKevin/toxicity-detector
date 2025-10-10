# Toxicity Detector

A multi-label text classification system for detecting toxic content using deep learning. The project implements two models (LSTM and BERT) trained on hate speech and offensive language datasets to classify content across 7 toxicity categories.

## Project Overview

This system classifies text into the following categories:
- `toxic`
- `severe_toxic`
- `obscene`
- `threat`
- `insult`
- `identity_hate`
- `non_offensive`

**Models:**
- BiLSTM with Attention (4.8M parameters)
- DistilBERT Fine-tuned (66M+ parameters)

**Performance:**
- LSTM: 80.5% ROC-AUC, 0.65 F1-Score
- BERT: 84.9% ROC-AUC, 0.64 F1-Score

## Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/OmondiKevin/toxicity-detector.git
cd toxicity-detector
```

### 2. Create and activate virtual environment

**Create virtual environment (all OS):**
```bash
python3 -m venv .venv
```

**Activate virtual environment:**

macOS/Linux:
```bash
source .venv/bin/activate
```

Windows (Command Prompt):
```cmd
.venv\Scripts\activate.bat
```

Windows (PowerShell):
```powershell
.venv\Scripts\Activate.ps1
```

**Note:** You should see `(.venv)` in your terminal prompt when activated.

### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download NLTK data
```bash
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### 5. Verify datasets
The repository includes both raw and processed datasets:
- `data/raw/` - Original datasets
- `data/processed/` - Pre-split train/val/test files

If you need to regenerate processed data:
```bash
make prepare-merged
make split-multilabel
```

### 6. Train models
**Note:** Model files are not committed to Git. You must train them locally.

Train LSTM model (~45 minutes on M1 CPU):
```bash
make train-lstm
```

Train BERT model (~4-5 hours on M1 CPU):
```bash
make train-bert
```

### 7. Evaluate models
```bash
make eval-multilabel
```
Outputs saved to `evaluation_results/` including metrics, confusion matrices, and comparison charts.

### 8. Run the API server
```bash
make api
```
Server runs at http://127.0.0.1:8000

Test the API:
```bash
curl -X POST http://127.0.0.1:8000/classify_multilabel_bert \
  -H 'Content-Type: application/json' \
  -d '{"text":"Example text","threshold":0.5}'
```

### 9. Launch Streamlit demo
```bash
make demo
```
Interactive web interface for testing the models.

## Makefile Targets

| Target | Description |
|--------|-------------|
| `prepare-merged` | Merge raw datasets into multilabel format |
| `split-multilabel` | Create train/val/test splits |
| `train-lstm` | Train LSTM model |
| `train-bert` | Train BERT model |
| `eval-multilabel` | Evaluate both models with metrics |
| `api` | Start FastAPI server |
| `demo` | Launch Streamlit UI |

## API Endpoints

- `GET /health` - Health check
- `GET /version` - Version information
- `POST /classify_multilabel_lstm` - Classify with LSTM model
- `POST /classify_multilabel_bert` - Classify with BERT model

## Project Structure

```
toxicity-detector/
├── data/
│   ├── raw/                    # Original datasets
│   └── processed/              # Train/val/test splits
├── models/                     # Trained models (git-ignored)
├── src/
│   ├── api.py                  # FastAPI endpoints
│   ├── model_lstm.py           # LSTM architecture
│   ├── model_bert.py           # BERT architecture
│   ├── train_lstm.py           # LSTM training script
│   ├── train_bert.py           # BERT training script
│   ├── evaluate_multilabel.py  # Evaluation & metrics
│   └── preprocess.py           # Text cleaning utilities
├── app/
│   └── streamlit_app_multilabel.py  # Streamlit demo
├── Makefile                    # Build automation
└── requirements.txt
```

## Model Artifacts

Trained model files are **not stored in Git** due to their size (100MB+). To use the API or demo:
1. Train models locally: `make train-lstm` and/or `make train-bert`
2. Models will be saved to `models/` directory
3. API and demo will load models from this directory

## Hardware Requirements

- Minimum: 8GB RAM, any CPU
- Recommended: 16GB+ RAM for faster training
- GPU support optional (speeds up training significantly)
- Developed on: Apple MacBook Pro M1, 8GB RAM, CPU-only

## Troubleshooting

**Virtual environment not active?**

macOS/Linux:
```bash
source .venv/bin/activate
```

Windows (Command Prompt):
```cmd
.venv\Scripts\activate.bat
```

Windows (PowerShell):
```powershell
.venv\Scripts\Activate.ps1
```

**Missing NLTK data?**
```bash
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

**API error: Model not found?**
Train models first:
```bash
make train-lstm
make train-bert
```

**Port already in use?**
```bash
uvicorn src.api:app --reload --port 8001
```

**Out of memory during training?**
Reduce batch size in `src/train_lstm.py` or `src/train_bert.py`.

---

## For Reviewers

**See [REVIEWER_GUIDE.md](REVIEWER_GUIDE.md)** for the simplest way to unzip, set up, and run the project without retraining.

The packaged `toxicity-detector.zip` includes:
- Trained models (BERT + LSTM) - ready to use
- All datasets (raw + processed)
- Evaluation results and visualizations
- Complete source code and documentation

No training required - just unzip, install dependencies, and run!
