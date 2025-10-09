# Toxicity Detector

A toxicity detection system with multiclass classification baseline and API service.

## Data Policy for This Repo

This project uses only the files provided under `data/raw/`:
- `hate_offensive_speech_detection.csv` (multi-class: hate/offensive/neutral)
- `sample_submission.csv` (multi-label template: toxic, severe_toxic, obscene, threat, insult, identity_hate)

No external downloads are required. The multilabel step is currently a stub that mirrors the sample submission format.

## Quick Start

### 1. Data Preparation
```bash
make prepare
```

### 2. Train Multiclass Baseline
```bash
make train-mc
```

### 3. Evaluate
```bash
make eval-mc
```

### 4. Run API Server
```bash
make api
```

### 5. Run Demo UI
```bash
make demo
```

### 6. Generate Submission (Stub)
```bash
make submit
```

## Available Targets

- `prepare` - Prepare multiclass training data
- `train-mc` - Train multiclass TF-IDF + SVM baseline
- `eval-mc` - Evaluate multiclass model
- `api` - Start FastAPI server
- `demo` - Launch Streamlit demo UI
- `submit` - Generate stub multilabel submission

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
│   └── multiclass_tfidf_svm.joblib
├── src/
│   ├── api.py                  # FastAPI service
│   ├── evaluate_multiclass.py  # Evaluation script
│   ├── infer_multilabel.py     # Multilabel stub
│   ├── prepare_multiclass.py   # Data preparation
│   ├── preprocess.py           # Text cleaning utilities
│   └── train_multiclass_baseline.py  # Training script
└── requirements.txt
```

## API Endpoints

- `GET /health` - Health check
- `GET /version` - Version info
- `POST /classify_multiclass` - Multiclass classification (hate/offensive/neutral)
- `POST /classify_multilabel` - Multilabel classification (requires model training)

## License

See LICENSE file for details.

