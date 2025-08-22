# src/utils_metrics.py
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_curve,
    average_precision_score
)

def save_jsonl(path: str, d: dict) -> None:
    """Append a json line to a log file (creates folders if needed)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          classes: list[str], out_path: str) -> None:
    """Save a confusion matrix figure (no explicit colors per instructions)."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    fig = plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

def plot_pr_curve(y_true_onehot: np.ndarray, y_score: np.ndarray,
                  classes: list[str], out_path: str) -> float:
    """
    Plot one-vs-rest PR curves for each class and return macro AP.
    y_true_onehot: (N, C) one-hot ground truth
    y_score: (N, C) probabilities
    """
    aps = []
    fig = plt.figure(figsize=(8, 6))
    for k in range(y_true_onehot.shape[1]):
        precision, recall, _ = precision_recall_curve(y_true_onehot[:, k], y_score[:, k])
        ap = average_precision_score(y_true_onehot[:, k], y_score[:, k])
        aps.append(ap)
        plt.plot(recall, precision, label=f"{classes[k]} (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"PR Curves (macro AP={float(np.mean(aps)):.3f})")
    plt.legend(ncol=2, fontsize=8)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return float(np.mean(aps))
 