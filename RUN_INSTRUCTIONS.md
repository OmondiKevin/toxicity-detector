# Run Instructions

## Prerequisites
- Python 3.10+ (3.11+ recommended)
- pip

## Quick Start

### 1. Unzip the Release Package
```bash
unzip toxicity-detector-v<version>.zip
cd toxicity-detector-v<version>
```

### 2. Create Virtual Environment (Recommended)
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Download NLTK Data (Required)
```bash
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### 5. Run the Application

**Option A: Streamlit Demo (Recommended)**
```bash
streamlit run app/streamlit_app_multilabel.py
```
The demo will open in your browser at `http://localhost:8501`

**Option B: FastAPI Server**
```bash
uvicorn src.api:app --reload --port 8000
```
API will be available at `http://127.0.0.1:8000`

Test the API:
```bash
curl -X POST http://127.0.0.1:8000/classify_multilabel_bert \
  -H "Content-Type: application/json" \
  -d '{"text":"Example text","threshold":0.5}'
```

## File Locations
- Models: `models/` directory (should contain `.pth` and `*config.json` files)
- Processed data: `data/processed/` directory (CSV files)
- Source code: `app/` and `src/` directories

## Troubleshooting

**Models not found?**
- Verify `models/` directory contains `.pth` files
- Check that the ZIP was extracted completely

**NLTK errors?**
- Run the NLTK download command in step 4

**Import errors?**
- Ensure virtual environment is activated
- Reinstall dependencies: `pip install -r requirements.txt`

For more details, see [README.md](README.md) or [REVIEWER_GUIDE.md](REVIEWER_GUIDE.md).

