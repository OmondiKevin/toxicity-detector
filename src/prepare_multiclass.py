import pandas as pd
from sklearn.model_selection import train_test_split
from preprocess import clean_text

LABEL_MAP = {1: "hate", 2: "offensive", 3: "neutral"}
RAW_PATH = "data/raw/hate_offensive_speech_detection.csv"
OUT_DIR = "data/processed"

def run(seed: int = 42):
    df = pd.read_csv(RAW_PATH)
    df = df.rename(columns={"tweet": "text"})
    df["text"] = df["text"].astype(str).map(clean_text)
    df["label_name"] = df["label"].map(LABEL_MAP)

    train_df, test_df = train_test_split(
        df, test_size=0.20, random_state=seed, stratify=df["label"]
    )
    train_df, val_df = train_test_split(
        train_df, test_size=0.125, random_state=seed, stratify=train_df["label"]
    )

    train_df.to_csv(f"{OUT_DIR}/train.csv", index=False)
    val_df.to_csv(f"{OUT_DIR}/val.csv", index=False)
    test_df.to_csv(f"{OUT_DIR}/test.csv", index=False)

if __name__ == "__main__":
    run()

