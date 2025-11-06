# Reviewer Guide

This guide helps reviewers quickly set up and test the toxicity detector package.

## Package Contents

The `toxicity-detector-v<version>.zip` package includes:
- **Trained models** under `models/` (BERT + LSTM, ready to use)
- **Processed datasets** under `data/processed/` (train/val/test splits)
- **Application code** under `app/` (Streamlit demo)
- **Source code** under `src/` (models, API, utilities)
- **Documentation** (README.md, RUN_INSTRUCTIONS.md, this file)
- **Scripts** under `scripts/` (optional utilities)

## Quick Setup (5 minutes)

### 1. Extract and Enter
```bash
unzip toxicity-detector-v<version>.zip
cd toxicity-detector-v<version>
```

### 2. Set Up Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Download NLTK Data
```bash
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### 4. Launch Application
```bash
streamlit run app/streamlit_app_multilabel.py
```

The Streamlit UI will open in your browser automatically.

## What to Verify

### 1. Application Launch
- ✅ Streamlit app loads without errors
- ✅ Models load successfully (check console for "Model loaded" messages)
- ✅ UI displays correctly with all controls visible

### 2. Basic Inference
Test with sample inputs:
- **Safe text**: "Thank you for your help!"
  - Expected: High `non_offensive` probability, low toxic probabilities
- **Toxic text**: "I hate this stupid thing"
  - Expected: High `toxic` or `insult` probability, low `non_offensive`
- **Severe text**: "You should die"
  - Expected: High `severe_toxic` or `threat` probability

### 3. Model Comparison
- Both LSTM and BERT models should produce predictions
- Results should be consistent across multiple runs with the same input
- Moderation actions (BLOCK/FLAG/ALLOW) should be reasonable

### 4. File Verification
Check that required files exist:
- `models/bert_multilabel.pth` and `models/lstm_multilabel.pth`
- `models/*config.json` files
- `data/processed/*.csv` files
- `app/streamlit_app_multilabel.py`
- `src/` directory with all modules

## Expected Outputs

- **UI loads** without additional configuration
- **Predictions are consistent** across runs (same input = same output)
- **Both models work** (LSTM and BERT)
- **Moderation actions** are appropriate (BLOCK for severe, FLAG for moderate, ALLOW for safe)

## Troubleshooting

**Models not loading?**
- Verify `models/` directory contains `.pth` files
- Check console for error messages
- Ensure all files were extracted from ZIP

**Import errors?**
- Activate virtual environment: `source .venv/bin/activate`
- Reinstall: `pip install -r requirements.txt`

**NLTK errors?**
- Run: `python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"`

For detailed setup instructions, see [RUN_INSTRUCTIONS.md](RUN_INSTRUCTIONS.md).
