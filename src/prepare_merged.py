"""
Prepare merged multilabel dataset combining:
- Text from hate_offensive_speech_detection.csv
- Label structure from sample_submission.csv schema

Maps 3-class labels to 7 multi-label categories using intelligent heuristics.
"""
import pandas as pd
from src.preprocess import clean_text_advanced

RAW_HATE_PATH = "data/raw/hate_offensive_speech_detection.csv"
OUT_DIR = "data/processed"
OUT_FILE = f"{OUT_DIR}/merged_multilabel.csv"

# Heuristic mapping: hate speech -> severe_toxic + identity_hate; offensive -> obscene
# Real annotations would improve performance significantly
LABEL_MAPPING = {
    1: {  # Hate speech
        "toxic": 1,
        "severe_toxic": 1,
        "obscene": 0,
        "threat": 0,
        "insult": 1,
        "identity_hate": 1,
        "non_offensive": 0
    },
    2: {  # Offensive language
        "toxic": 1,
        "severe_toxic": 0,
        "obscene": 1,
        "threat": 0,
        "insult": 1,
        "identity_hate": 0,
        "non_offensive": 0
    },
    3: {  # Neutral/Neither
        "toxic": 0,
        "severe_toxic": 0,
        "obscene": 0,
        "threat": 0,
        "insult": 0,
        "identity_hate": 0,
        "non_offensive": 1
    }
}


def run(lemmatize: bool = True):
    """
    Merge datasets and create multilabel structure.

    Args:
        lemmatize: Apply lemmatization during text cleaning
    """
    print("=" * 80)
    print("DATA PREPROCESSING - MERGING DATASETS")
    print("=" * 80)

    # Load data
    print(f"\nLoading {RAW_HATE_PATH}...")
    df = pd.read_csv(RAW_HATE_PATH)
    print(f"Loaded {len(df):,} samples")

    print(f"\nCleaning text (URLs, mentions, hashtags, emojis, punctuation, lemmatization={lemmatize})...")
    df['text'] = df['tweet'].astype(str).apply(lambda x: clean_text_advanced(x, lemmatize=lemmatize))

    original_len = len(df)
    df = df[df['text'].str.strip() != '']
    if len(df) < original_len:
        print(f"Removed {original_len - len(df)} empty texts")

    print("\nMapping 3-class labels to 7-class multi-label structure...")

    for col in ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate", "non_offensive"]:
        df[col] = df['label'].map(lambda x: LABEL_MAPPING[x][col])

    output_columns = ["text", "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate", "non_offensive"]
    df_out = df[output_columns].copy()

    print(f"\nSaving to {OUT_FILE}...")
    df_out.to_csv(OUT_FILE, index=False)
    print(f"Shape: {df_out.shape}")

    print("\nLabel distribution:")
    for col in ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate", "non_offensive"]:
        positive = df_out[col].sum()
        pct = 100 * positive / len(df_out)
        print(f"  {col:20s}: {positive:5d} ({pct:5.1f}%)")

    print("\n" + "=" * 80)
    print("Preprocessing completed")
    print("=" * 80)

    return df_out


if __name__ == "__main__":
    run(lemmatize=True)
