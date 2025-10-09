import pandas as pd

# This stub only ensures pipeline wiring using the provided sample_submission.csv.
# It does NOT train a multilabel model (no multilabel train data provided).
# It copies the template and can optionally apply a simple mapping from the multiclass model, if desired, later.

IN_PATH = "data/raw/sample_submission.csv"
OUT_PATH = "preds_sample_submission.csv"


def main():
    sub = pd.read_csv(IN_PATH)
    # Keep default 0.5 probabilities to prove end-to-end pipeline; replace later with heuristic or model outputs.
    sub.to_csv(OUT_PATH, index=False)
    print(f"Wrote stub multilabel submission to {OUT_PATH}")


if __name__ == "__main__":
    main()
