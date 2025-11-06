"""
Split merged multilabel dataset into train/val/test sets.
Uses stratified split based on label combinations for better distribution.
"""
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_FILE = "data/processed/merged_multilabel.csv"
OUT_DIR = "data/processed"
RANDOM_SEED = 42

LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate", "non_offensive"]


def create_stratify_column(df, label_cols):
    """
    Create a stratification column based on label combinations.
    For multi-label, we use the most dominant label for stratification.
    """
    # Use the most frequent label combination or primary label
    # For simplicity, we'll stratify on 'non_offensive' as it's the most balanced
    return df['non_offensive']


def run(test_size=0.20, val_size=0.125, seed=RANDOM_SEED):
    """
    Split dataset into train/val/test.

    Args:
        test_size: Proportion for test set (from total)
        val_size: Proportion for validation set (from train)
        seed: Random seed for reproducibility
    """
    print("=" * 80)
    print("MULTILABEL DATASET SPLITTING")
    print("=" * 80)

    # Load merged dataset
    print(f"\n1. Loading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    print(f"   Total samples: {len(df):,}")
    print(f"   Columns: {list(df.columns)}")

    # Create stratification column
    print("\n2. Creating stratified split...")
    stratify_col = create_stratify_column(df, LABEL_COLS)

    # First split: train+val vs test
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=stratify_col
    )

    print(f"   Train+Val: {len(train_val_df):,} samples")
    print(f"   Test: {len(test_df):,} samples")

    # Second split: train vs val
    stratify_col_train = create_stratify_column(train_val_df, LABEL_COLS)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size,
        random_state=seed,
        stratify=stratify_col_train
    )

    print(f"   Train: {len(train_df):,} samples")
    print(f"   Val: {len(val_df):,} samples")

    # Save splits
    print("\n3. Saving splits...")
    train_path = f"{OUT_DIR}/train_multilabel.csv"
    val_path = f"{OUT_DIR}/val_multilabel.csv"
    test_path = f"{OUT_DIR}/test_multilabel.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"   Saved: {train_path}")
    print(f"   Saved: {val_path}")
    print(f"   Saved: {test_path}")

    # Show label distribution for each split
    print("\n4. Label distribution across splits:")
    for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        print(f"\n   {split_name} ({len(split_df):,} samples):")
        for col in LABEL_COLS:
            count = split_df[col].sum()
            pct = 100 * count / len(split_df)
            print(f"     {col:20s}: {count:5d} ({pct:5.1f}%)")

    print("\n" + "=" * 80)
    print("Dataset split completed")
    print("=" * 80)

    return train_df, val_df, test_df


if __name__ == "__main__":
    run()
