# Multilabel Setup (Current Status)

## Overview

The current repository includes a multilabel submission stub that uses `data/raw/sample_submission.csv` as a format template.

## Current Implementation

- **Input**: `data/raw/sample_submission.csv` (template with 6 toxicity labels)
- **Output**: `preds_sample_submission.csv` (stub with 0.5 probabilities)
- **Status**: Stub implementation only

## What This Means

- We only use `data/raw/sample_submission.csv` as a format template
- No multilabel training data is provided in this repo
- The current pipeline writes a stub `preds_sample_submission.csv` with default 0.5 probabilities
- This ensures the end-to-end pipeline is wired correctly

## Future Extensions

When multilabel training data becomes available in `data/raw/`, you can:

1. Add multilabel training data
2. Implement a real multilabel model (e.g., BERT with sigmoid outputs)
3. Replace the stub in `src/infer_multilabel.py` with actual model inference
4. Generate real probability predictions

## Running the Stub

Generate the stub submission:

```bash
make submit
```

This creates `preds_sample_submission.csv` with the same format as the template.

## Available Data

The repository currently works with:
- **Multiclass data**: `hate_offensive_speech_detection.csv` (for hate/offensive/neutral classification)
- **Multilabel template**: `sample_submission.csv` (format reference only)

No external dataset downloads are required for the current functionality.
