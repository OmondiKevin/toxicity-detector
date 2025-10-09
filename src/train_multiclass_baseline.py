import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from preprocess import clean_text

TRAIN_CSV = "data/processed/train.csv"
VAL_CSV = "data/processed/val.csv"  # (not used for SVM training but reserved for future tuning)
MODEL_OUT = "models/multiclass_tfidf_svm.joblib"

# Optional: readable mapping if needed elsewhere
LABEL_MAP = {1: "hate", 2: "offensive", 3: "neutral"}


def build_pipeline():
    # Char-ngrams handle misspellings/slurs well
    tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3, max_features=200_000)
    clf = LinearSVC()
    return Pipeline([("tfidf", tfidf), ("clf", clf)])


def main():
    tr = pd.read_csv(TRAIN_CSV)
    # Safety: (re)clean if needed
    tr["x"] = tr["text"].astype(str).map(clean_text)
    y = tr["label"].values

    pipe = build_pipeline()
    pipe.fit(tr["x"], y)

    joblib.dump({
        "pipeline": pipe,
        "classes_": sorted(tr["label"].unique().tolist()),
        "label_map": LABEL_MAP
    }, MODEL_OUT)
    print(f"Saved model to {MODEL_OUT}")


if __name__ == "__main__":
    main()

