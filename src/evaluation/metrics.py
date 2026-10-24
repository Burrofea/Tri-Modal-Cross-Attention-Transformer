"""Comprehensive evaluation metrics for deepfake detection."""
from typing import Dict, Tuple
import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, average_precision_score,
    confusion_matrix
)

class EvaluatorMetrics:
    """
    Computes statistical and clinical performance metrics for binary and multi-task predictions.
    """
    @staticmethod
    def find_optimal_threshold(y_true: np.ndarray, y_probs: np.ndarray) -> float:
        """
        Finds optimal classification threshold on validation set maximizing balanced accuracy / F1.
        """
        thresholds = np.linspace(0.1, 0.9, 81)
        best_f1, best_th = -1.0, 0.5
        for th in thresholds:
            preds = (y_probs >= th).astype(int)
            f1 = f1_score(y_true, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_th = th
        return float(best_th)

    @classmethod
    def compute_binary_metrics(
        cls,
        y_true: np.ndarray,
        y_probs: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """
        Computes accuracy, balanced accuracy, precision, recall, F1, ROC-AUC, and PR-AUC.
        """
        preds = (y_probs >= threshold).astype(int)

        try:
            auc = float(roc_auc_score(y_true, y_probs))
        except Exception:
            auc = 0.5

        try:
            pr_auc = float(average_precision_score(y_true, y_probs))
        except Exception:
            pr_auc = float(np.mean(y_true))

        acc = float(accuracy_score(y_true, preds))
        b_acc = float(balanced_accuracy_score(y_true, preds))
        prec = float(precision_score(y_true, preds, zero_division=0))
        rec = float(recall_score(y_true, preds, zero_division=0))
        f1 = float(f1_score(y_true, preds, zero_division=0))

        # Expected Calibration Error (ECE) approximation
        bin_edges = np.linspace(0.0, 1.0, 11)
        ece = 0.0
        for i in range(len(bin_edges) - 1):
            mask = (y_probs >= bin_edges[i]) & (y_probs < bin_edges[i + 1])
            if np.any(mask):
                acc_bin = np.mean(y_true[mask])
                conf_bin = np.mean(y_probs[mask])
                ece += np.sum(mask) * np.abs(acc_bin - conf_bin)
        ece /= max(1, len(y_probs))

        return {
            "accuracy": acc,
            "balanced_accuracy": b_acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": auc,
            "pr_auc": pr_auc,
            "ece": float(ece),
            "threshold": threshold,
        }
