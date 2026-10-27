"""Evaluation metrics utilities for toxicity classification models."""

from typing import Any, Dict, List, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)


def evaluate_classifier_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.50
) -> Dict[str, Any]:
    """Compute comprehensive evaluation metrics for a binary classifier.
    
    Returns:
        Dict with precision, recall, F1 per class, macro F1, PR-AUC, ROC-AUC,
        confusion matrix, accuracy, and support counts.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    # Class-wise metrics (0: Non-Toxic, 1: Toxic)
    p_per_class, r_per_class, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    accuracy = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    # PR-AUC
    precision_pts, recall_pts, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall_pts, precision_pts)

    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = float("nan")

    return {
        "accuracy": float(accuracy),
        "threshold": float(threshold),
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "macro_f1": float(macro_f1),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "non_toxic": {
            "precision": float(p_per_class[0]),
            "recall": float(r_per_class[0]),
            "f1": float(f1_per_class[0]),
            "support": int(support[0]),
        },
        "toxic": {
            "precision": float(p_per_class[1]),
            "recall": float(r_per_class[1]),
            "f1": float(f1_per_class[1]),
            "support": int(support[1]),
        },
        "confusion_matrix": cm.tolist(),
        "true_negatives": int(cm[0, 0]),
        "false_positives": int(cm[0, 1]),
        "false_negatives": int(cm[1, 0]),
        "true_positives": int(cm[1, 1]),
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: List[float] = None
) -> List[Dict[str, Any]]:
    """Evaluate performance across an array of decision thresholds."""
    if thresholds is None:
        thresholds = [0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.70, 0.80]

    sweep_results = []
    for th in thresholds:
        metrics = evaluate_classifier_predictions(y_true, y_prob, threshold=th)
        sweep_results.append(metrics)
    return sweep_results

