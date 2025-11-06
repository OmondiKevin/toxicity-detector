"""
Evaluate multilabel classification models.
Computes metrics and generates visualizations for LSTM and BERT models.
"""
import os
import json

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_fscore_support,
    roc_auc_score,
    multilabel_confusion_matrix
)

from src.model_lstm import LSTMMultilabelClassifier
from src.model_bert import BERTMultilabelClassifier
from src.dataset_utils import create_lstm_dataloaders, create_bert_dataloaders

TEST_PATH = "data/processed/test_multilabel.csv"
LSTM_MODEL_PATH = "models/lstm_multilabel.pth"
LSTM_VOCAB_PATH = "models/lstm_vocab.pth"
LSTM_CONFIG_PATH = "models/lstm_config.json"
BERT_MODEL_PATH = "models/bert_multilabel.pth"
BERT_CONFIG_PATH = "models/bert_config.json"
OUTPUT_DIR = "evaluation_results"

LABEL_NAMES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate", "non_offensive"]


def evaluate_model(model, dataloader, device, model_name):
    """
    Evaluate a single model and return predictions and true labels.
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            if model_name == "LSTM":
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids)
            else:  # BERT
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask)

            # Store probabilities and predictions
            all_probs.append(outputs.cpu().numpy())
            all_preds.append((outputs > 0.5).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_probs = np.vstack(all_probs)
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    return all_probs, all_preds, all_labels


def compute_metrics(y_true, y_pred, y_probs):
    """
    Compute comprehensive metrics for multilabel classification.
    """
    metrics = {}

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='micro', zero_division=0
    )

    try:
        roc_auc_per_class = []
        for i in range(y_true.shape[1]):
            if len(np.unique(y_true[:, i])) > 1:  # Need both classes present
                auc = roc_auc_score(y_true[:, i], y_probs[:, i])
                roc_auc_per_class.append(auc)
            else:
                roc_auc_per_class.append(np.nan)
        roc_auc_macro = np.nanmean(roc_auc_per_class)
    except (ValueError, IndexError):
        roc_auc_per_class = [np.nan] * y_true.shape[1]
        roc_auc_macro = np.nan

    metrics['per_class'] = {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'roc_auc': roc_auc_per_class
    }

    metrics['macro'] = {
        'precision': precision_macro,
        'recall': recall_macro,
        'f1': f1_macro,
        'roc_auc': roc_auc_macro
    }

    metrics['micro'] = {
        'precision': precision_micro,
        'recall': recall_micro,
        'f1': f1_micro
    }

    return metrics


def plot_confusion_matrices(y_true, y_pred, label_names, model_name, output_dir):
    """
    Plot confusion matrix for each label.
    """
    cm_multilabel = multilabel_confusion_matrix(y_true, y_pred)

    n_labels = len(label_names)
    n_cols = 3
    n_rows = (n_labels + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4))
    axes = axes.flatten()

    for idx, (cm, label) in enumerate(zip(cm_multilabel, label_names)):
        ax = axes[idx]
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'])
        ax.set_title(f'{label.replace("_", " ").title()}')
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')

    for idx in range(n_labels, len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f'{model_name} - Confusion Matrices per Label', fontsize=16, y=1.0)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_name.lower()}_confusion_matrices.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_metrics_comparison(metrics_lstm, metrics_bert, label_names, output_dir):
    """
    Plot comparison of metrics between LSTM and BERT.
    """
    metrics_to_plot = ['precision', 'recall', 'f1', 'roc_auc']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for idx, metric_name in enumerate(metrics_to_plot):
        ax = axes[idx]

        lstm_values = metrics_lstm['per_class'][metric_name]
        bert_values = metrics_bert['per_class'][metric_name]

        x = np.arange(len(label_names))
        width = 0.35

        ax.bar(x - width / 2, lstm_values, width, label='LSTM', alpha=0.8)
        ax.bar(x + width / 2, bert_values, width, label='BERT', alpha=0.8)

        ax.set_xlabel('Labels')
        ax.set_ylabel(metric_name.upper().replace('_', '-'))
        ax.set_title(f'{metric_name.upper().replace("_", "-")} Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([label.replace('_', '\n') for label in label_names], rotation=0, ha='center')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 1.0])

    plt.suptitle('LSTM vs BERT - Performance Comparison', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/model_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()


def generate_report(metrics_lstm, metrics_bert, label_names, output_dir):
    """
    Generate a detailed text report.
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("MULTILABEL CLASSIFICATION - MODEL EVALUATION REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")

    report_lines.append("LSTM MODEL RESULTS")
    report_lines.append("-" * 80)
    report_lines.append(f"{'Label':<20} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'ROC-AUC':>10} {'Support':>10}")
    report_lines.append("-" * 80)

    for i, label in enumerate(label_names):
        p = metrics_lstm['per_class']['precision'][i]
        r = metrics_lstm['per_class']['recall'][i]
        f1 = metrics_lstm['per_class']['f1'][i]
        auc = metrics_lstm['per_class']['roc_auc'][i]
        sup = int(metrics_lstm['per_class']['support'][i])

        report_lines.append(f"{label:<20} {p:>10.4f} {r:>10.4f} {f1:>10.4f} {auc:>10.4f} {sup:>10d}")

    report_lines.append("-" * 80)
    report_lines.append(f"{'Macro Avg':<20} {metrics_lstm['macro']['precision']:>10.4f} "
                        f"{metrics_lstm['macro']['recall']:>10.4f} "
                        f"{metrics_lstm['macro']['f1']:>10.4f} "
                        f"{metrics_lstm['macro']['roc_auc']:>10.4f}")
    report_lines.append(f"{'Micro Avg':<20} {metrics_lstm['micro']['precision']:>10.4f} "
                        f"{metrics_lstm['micro']['recall']:>10.4f} "
                        f"{metrics_lstm['micro']['f1']:>10.4f}")
    report_lines.append("")

    # BERT Results
    report_lines.append("BERT MODEL RESULTS")
    report_lines.append("-" * 80)
    report_lines.append(f"{'Label':<20} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'ROC-AUC':>10} {'Support':>10}")
    report_lines.append("-" * 80)

    for i, label in enumerate(label_names):
        p = metrics_bert['per_class']['precision'][i]
        r = metrics_bert['per_class']['recall'][i]
        f1 = metrics_bert['per_class']['f1'][i]
        auc = metrics_bert['per_class']['roc_auc'][i]
        sup = int(metrics_bert['per_class']['support'][i])

        report_lines.append(f"{label:<20} {p:>10.4f} {r:>10.4f} {f1:>10.4f} {auc:>10.4f} {sup:>10d}")

    report_lines.append("-" * 80)
    report_lines.append(f"{'Macro Avg':<20} {metrics_bert['macro']['precision']:>10.4f} "
                        f"{metrics_bert['macro']['recall']:>10.4f} "
                        f"{metrics_bert['macro']['f1']:>10.4f} "
                        f"{metrics_bert['macro']['roc_auc']:>10.4f}")
    report_lines.append(f"{'Micro Avg':<20} {metrics_bert['micro']['precision']:>10.4f} "
                        f"{metrics_bert['micro']['recall']:>10.4f} "
                        f"{metrics_bert['micro']['f1']:>10.4f}")
    report_lines.append("")

    # Comparison
    report_lines.append("MODEL COMPARISON SUMMARY")
    report_lines.append("-" * 80)
    report_lines.append(f"{'Metric':<30} {'LSTM':>15} {'BERT':>15} {'Winner':>15}")
    report_lines.append("-" * 80)

    comparisons = [
        ('Macro F1-Score', metrics_lstm['macro']['f1'], metrics_bert['macro']['f1']),
        ('Macro Precision', metrics_lstm['macro']['precision'], metrics_bert['macro']['precision']),
        ('Macro Recall', metrics_lstm['macro']['recall'], metrics_bert['macro']['recall']),
        ('Macro ROC-AUC', metrics_lstm['macro']['roc_auc'], metrics_bert['macro']['roc_auc']),
        ('Micro F1-Score', metrics_lstm['micro']['f1'], metrics_bert['micro']['f1']),
    ]

    for metric_name, lstm_val, bert_val in comparisons:
        winner = "LSTM" if lstm_val > bert_val else "BERT" if bert_val > lstm_val else "TIE"
        report_lines.append(f"{metric_name:<30} {lstm_val:>15.4f} {bert_val:>15.4f} {winner:>15}")

    report_lines.append("=" * 80)

    report_text = "\n".join(report_lines)
    with open(f"{output_dir}/evaluation_report.txt", 'w') as f:
        f.write(report_text)

    return report_text


def run_evaluation():
    """Main evaluation function."""
    print("=" * 80)
    print("MODEL EVALUATION")
    print("=" * 80)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    print("\nEvaluating LSTM model...")

    with open(LSTM_CONFIG_PATH, 'r') as f:
        lstm_config = json.load(f)

    vocab = torch.load(LSTM_VOCAB_PATH, weights_only=False)
    _, _, lstm_test_loader, _ = create_lstm_dataloaders(
        "data/processed/train_multilabel.csv",
        "data/processed/val_multilabel.csv",
        TEST_PATH,
        vocab=vocab,
        max_length=lstm_config['max_length'],
        batch_size=lstm_config['batch_size']
    )

    lstm_model = LSTMMultilabelClassifier(
        vocab_size=len(vocab),
        embedding_dim=lstm_config['embedding_dim'],
        hidden_dim=lstm_config['hidden_dim'],
        num_layers=lstm_config['num_layers'],
        dropout=lstm_config['dropout'],
        num_labels=7
    ).to(device)

    checkpoint = torch.load(LSTM_MODEL_PATH, map_location=device, weights_only=False)
    lstm_model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded LSTM (val loss: {checkpoint['val_loss']:.4f})")

    lstm_probs, lstm_preds, lstm_labels = evaluate_model(lstm_model, lstm_test_loader, device, "LSTM")

    print("\nEvaluating BERT model...")

    with open(BERT_CONFIG_PATH, 'r') as f:
        bert_config = json.load(f)

    _, _, bert_test_loader, tokenizer = create_bert_dataloaders(
        "data/processed/train_multilabel.csv",
        "data/processed/val_multilabel.csv",
        TEST_PATH,
        model_name=bert_config['model_name'],
        max_length=bert_config['max_length'],
        batch_size=bert_config['batch_size']
    )

    bert_model = BERTMultilabelClassifier(
        num_labels=7,
        dropout=bert_config['dropout'],
        pretrained_model=bert_config['model_name']
    ).to(device)

    checkpoint = torch.load(BERT_MODEL_PATH, map_location=device, weights_only=False)
    bert_model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded BERT (val loss: {checkpoint['val_loss']:.4f})")

    bert_probs, bert_preds, bert_labels = evaluate_model(bert_model, bert_test_loader, device, "BERT")

    print("\nComputing metrics...")
    metrics_lstm = compute_metrics(lstm_labels, lstm_preds, lstm_probs)
    metrics_bert = compute_metrics(bert_labels, bert_preds, bert_probs)

    print("Generating visualizations...")
    plot_confusion_matrices(lstm_labels, lstm_preds, LABEL_NAMES, "LSTM", OUTPUT_DIR)
    plot_confusion_matrices(bert_labels, bert_preds, LABEL_NAMES, "BERT", OUTPUT_DIR)
    plot_metrics_comparison(metrics_lstm, metrics_bert, LABEL_NAMES, OUTPUT_DIR)

    print("\nGenerating report...")
    report = generate_report(metrics_lstm, metrics_bert, LABEL_NAMES, OUTPUT_DIR)

    def safe_float(value):
        """Convert to float, replacing NaN with None for valid JSON."""
        val = float(value)
        return None if np.isnan(val) else val

    results = {
        'lstm': {
            'macro': {
                'precision': safe_float(metrics_lstm['macro']['precision']),
                'recall': safe_float(metrics_lstm['macro']['recall']),
                'f1': safe_float(metrics_lstm['macro']['f1']),
                'roc_auc': safe_float(metrics_lstm['macro']['roc_auc'])
            },
            'micro': {
                'precision': safe_float(metrics_lstm['micro']['precision']),
                'recall': safe_float(metrics_lstm['micro']['recall']),
                'f1': safe_float(metrics_lstm['micro']['f1'])
            },
            'per_class': {
                label: {
                    'precision': safe_float(metrics_lstm['per_class']['precision'][i]),
                    'recall': safe_float(metrics_lstm['per_class']['recall'][i]),
                    'f1': safe_float(metrics_lstm['per_class']['f1'][i]),
                    'roc_auc': safe_float(metrics_lstm['per_class']['roc_auc'][i]),
                    'support': int(metrics_lstm['per_class']['support'][i])
                }
                for i, label in enumerate(LABEL_NAMES)
            }
        },
        'bert': {
            'macro': {
                'precision': safe_float(metrics_bert['macro']['precision']),
                'recall': safe_float(metrics_bert['macro']['recall']),
                'f1': safe_float(metrics_bert['macro']['f1']),
                'roc_auc': safe_float(metrics_bert['macro']['roc_auc'])
            },
            'micro': {
                'precision': safe_float(metrics_bert['micro']['precision']),
                'recall': safe_float(metrics_bert['micro']['recall']),
                'f1': safe_float(metrics_bert['micro']['f1'])
            },
            'per_class': {
                label: {
                    'precision': safe_float(metrics_bert['per_class']['precision'][i]),
                    'recall': safe_float(metrics_bert['per_class']['recall'][i]),
                    'f1': safe_float(metrics_bert['per_class']['f1'][i]),
                    'roc_auc': safe_float(metrics_bert['per_class']['roc_auc'][i]),
                    'support': int(metrics_bert['per_class']['support'][i])
                }
                for i, label in enumerate(LABEL_NAMES)
            }
        }
    }

    with open(f"{OUTPUT_DIR}/metrics.json", 'w') as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETED")
    print(f"Results saved to: {OUTPUT_DIR}/")
    print("=" * 80)

    print("\n" + report)

    return results


if __name__ == "__main__":
    run_evaluation()
