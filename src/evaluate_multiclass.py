import joblib
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

MODEL_IN = "models/multiclass_tfidf_svm.joblib"
TEST_CSV = "data/processed/test.csv"


def main():
    bundle = joblib.load(MODEL_IN)
    pipe = bundle["pipeline"]
    label_map = bundle.get("label_map", {1: "hate", 2: "offensive", 3: "neutral"})

    df = pd.read_csv(TEST_CSV)
    X = df["text"].astype(str)
    y_true = df["label"].values
    y_pred = pipe.predict(X)

    print("Label names:", label_map)
    print(classification_report(y_true, y_pred, digits=4))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()

