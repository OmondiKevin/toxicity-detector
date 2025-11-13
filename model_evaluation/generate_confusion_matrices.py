"""
Generate confusion matrix visualizations for the saved multilabel model predictions.

This script loads the persisted test set labels and prediction scores for both the LSTM
and BERT models, thresholds the scores to obtain binary predictions, and then plots
per-class confusion matrices in a grid layout. The resulting PNG files are saved in
`model_evaluation/` to satisfy the interview deliverable that asks for visualizations
of confusion matrices for each class.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Dict, Tuple

import matplotlib

matplotlib.use("Agg")  # Use a non-interactive backend for headless environments.

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix


LABEL_NAMES = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
    "non_offensive",
]

MODEL_SOURCES: Dict[str, Dict[str, str]] = {
    "lstm": {
        "labels": "models/lstm_test_labels.npy",
        "preds": "models/lstm_test_preds.npy",
    },
    "bert": {
        "labels": "models/bert_test_labels.npy",
        "preds": "models/bert_test_preds.npy",
    },
}

OUTPUT_DIR = Path("model_evaluation")
THRESHOLD = 0.5


def load_arrays(paths: Dict[str, str]) -> Tuple[np.ndarray, np.ndarray]:
    """Load ground-truth labels and prediction scores from disk."""
    labels_path = Path(paths["labels"])
    preds_path = Path(paths["preds"])

    if not labels_path.exists():
        raise FileNotFoundError(f"Expected labels file not found: {labels_path}")
    if not preds_path.exists():
        raise FileNotFoundError(f"Expected predictions file not found: {preds_path}")

    labels = np.load(labels_path)
    preds = np.load(preds_path)

    if labels.shape != preds.shape:
        raise ValueError(
            f"Shape mismatch between labels {labels.shape} and predictions {preds.shape}"
        )

    return labels, preds


def plot_confusion_matrices(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    model_name: str,
    output_dir: Path,
    threshold: float = THRESHOLD,
) -> Path:
    """
    Plot confusion matrices for each label and save as a grid image.

    Returns the path to the generated PNG file.
    """
    y_pred = (y_scores >= threshold).astype(int)
    matrices = multilabel_confusion_matrix(y_true, y_pred)

    n_labels = len(LABEL_NAMES)
    n_cols = min(3, n_labels)
    n_rows = math.ceil(n_labels / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

    # When n_rows * n_cols == 1, axes is not an array—normalize to array for iteration.
    if isinstance(axes, plt.Axes):
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (matrix, label) in enumerate(zip(matrices, LABEL_NAMES)):
        ax = axes[idx]
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=ax,
            xticklabels=["Pred 0", "Pred 1"],
            yticklabels=["True 0", "True 1"],
        )
        ax.set_title(label.replace("_", " ").title())
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

    # Hide any unused subplots.
    for ax in axes[len(matrices) :]:
        ax.axis("off")

    plt.suptitle(f"{model_name.upper()} - Confusion Matrices per Class", fontsize=18)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    output_path = output_dir / f"{model_name.lower()}_confusion_matrices.png"
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    generated_files = []
    for model_name, paths in MODEL_SOURCES.items():
        labels, scores = load_arrays(paths)
        output_path = plot_confusion_matrices(labels, scores, model_name, OUTPUT_DIR)
        generated_files.append(output_path)
        print(f"[OK] Saved confusion matrices for {model_name.upper()} to {output_path}")

    print("\nGeneration complete. Files created:")
    for path in generated_files:
        print(f" - {path}")


if __name__ == "__main__":
    main()


