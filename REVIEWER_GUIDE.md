# Reviewer Guide

This project ships with **trained models** in the `models/` directory so you do not need to retrain (saves 4-5 hours).

## Quick Start (5 minutes)

### 1. Unzip & Enter
```bash
unzip toxicity-detector.zip
cd toxicity-detector
```

### 2. Create & Activate Environment

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (Command Prompt):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download NLTK Data
```bash
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### 5. Verify Required Files

The following files should be present (already included in the zip):

**Data:**
- `data/raw/hate_offensive_speech_detection.csv`
- `data/raw/sample_submission.csv`
- `data/processed/` (train/val/test splits)

**Trained Models (included - no training needed):**
- `models/bert_multilabel.pth` (762MB)
- `models/lstm_multilabel.pth` (55MB)
- `models/bert_config.json`
- `models/lstm_config.json`
- `models/lstm_vocab.pth`
- `models/*_test_labels.npy` and `*_test_preds.npy`

**Evaluation Results:**
- `evaluation_results/metrics.json`
- `evaluation_results/evaluation_report.txt`
- `evaluation_results/lstm_confusion_matrices.png`
- `evaluation_results/bert_confusion_matrices.png`
- `evaluation_results/model_comparison.png`

## Running the Project

**Tip:** Run `make help` to see all available Makefile commands.

### Option A: FastAPI Server

Start the API server:
```bash
uvicorn src.api:app --reload --port 8000
```

Or use the Makefile:
```bash
make api
```

**Test the API:**

Health check:
```bash
curl http://127.0.0.1:8000/health
```

LSTM Classification:
```bash
curl -X POST http://127.0.0.1:8000/classify_multilabel_lstm \
  -H "Content-Type: application/json" \
  -d '{"text":"I hate you","threshold":0.5}'
```

BERT Classification:
```bash
curl -X POST http://127.0.0.1:8000/classify_multilabel_bert \
  -H "Content-Type: application/json" \
  -d '{"text":"Have a nice day","threshold":0.5}'
```

**API Endpoints:**
- `GET /health` - Health check
- `GET /version` - Version info
- `POST /classify_multilabel_lstm` - Classify with LSTM model
- `POST /classify_multilabel_bert` - Classify with BERT model

### Option B: Streamlit Demo (Recommended)

Launch the interactive demo:
```bash
streamlit run app/streamlit_app_multilabel.py
```

Or use the Makefile:
```bash
make demo
```

The Streamlit UI will open in your browser with:
- Real-time text classification
- Visual probability breakdown
- Side-by-side model comparison (LSTM vs BERT)
- Moderation recommendations (BLOCK/FLAG/ALLOW)
- Interactive threshold adjustment

### Option C: Evaluation (Review Model Performance)

To regenerate evaluation metrics and visualizations:
```bash
make eval-multilabel
```

Results will be saved to `evaluation_results/`:
- `metrics.json` - Machine-readable metrics
- `evaluation_report.txt` - Detailed performance report
- `lstm_confusion_matrices.png` - LSTM confusion matrices (7 classes)
- `bert_confusion_matrices.png` - BERT confusion matrices (7 classes)
- `model_comparison.png` - Side-by-side comparison charts

## What You Get

### Pre-trained Models

**LSTM Model:**
- Architecture: BiLSTM with Attention
- Parameters: 4.8M
- Performance: 80.5% ROC-AUC, 0.65 F1-Score
- Training time: 45 minutes (already done for you)

**BERT Model:**
- Architecture: DistilBERT Fine-tuned
- Parameters: 66M+
- Performance: 84.9% ROC-AUC, 0.64 F1-Score
- Training time: 4-5 hours (already done for you)

### Classification Categories

The models classify text into 7 toxicity categories:
1. `toxic` - General toxicity
2. `severe_toxic` - Severe toxicity
3. `obscene` - Obscene language
4. `threat` - Threatening language
5. `insult` - Insults
6. `identity_hate` - Identity-based hate speech
7. `non_offensive` - Non-offensive content

## Troubleshooting

**Virtual environment not active?**
You should see `(.venv)` in your prompt. Re-activate:
- macOS/Linux: `source .venv/bin/activate`
- Windows CMD: `.venv\Scripts\activate.bat`
- Windows PS: `.venv\Scripts\Activate.ps1`

**Missing NLTK data?**
```bash
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

**API error: Model not found?**
Verify the `models/` directory contains `.pth` files. If missing, the zip extraction may have failed.

**Port already in use?**
```bash
uvicorn src.api:app --reload --port 8001
```

**Out of memory?**
The BERT model requires ~2GB RAM for inference. Use LSTM endpoint if memory constrained.

## Documentation

- **README.md** - Complete setup and usage guide
- **INTERVIEW_REPORT.md** - Comprehensive technical report (11KB, 359 lines)
  - Approach and methodology
  - Model architectures
  - Performance analysis
  - Challenges faced
  - Production recommendations
- **requirements.txt** - All dependencies with pinned versions
- **Makefile** - Automation targets (run `make help` to see all commands)

## Notes

- **No training required** - Models are pre-trained and included
- **Runs fully offline** - All required files included in the zip
- **Reproducible** - All dependencies pinned to specific versions
- **Cross-platform** - Works on macOS, Linux, and Windows
- **Production-ready** - FastAPI + Streamlit deployment included

## Performance Benchmarks

**System Used:** Apple MacBook Pro M1, 8GB RAM (CPU-only)

**Inference Times:**
- LSTM: ~50ms per request
- BERT: ~1s per request

**Model Performance:**
- LSTM F1-Score: 0.65 (better balance)
- BERT ROC-AUC: 0.85 (better discrimination)

## Need to Retrain?

If you want to retrain models from scratch (not necessary):

```bash
# View all available commands
make help

# Regenerate data splits (if needed)
make prepare-merged
make split-multilabel

# Train LSTM (~45 min on M1 CPU)
make train-lstm

# Train BERT (~4-5 hours on M1 CPU)
make train-bert

# Evaluate both models
make eval-multilabel
```

## Support

For questions or issues, refer to:
1. **INTERVIEW_REPORT.md** - Detailed technical documentation
2. **README.md** - Setup and usage guide
3. GitHub Issues (if repository is public)

---

**Enjoy reviewing the project!** Everything is pre-configured and ready to run.

