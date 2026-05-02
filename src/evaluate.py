"""
evaluate.py – Model evaluation utilities.

Usage
-----
from src.evaluate import evaluate_model, plot_confusion_matrix, plot_feature_importance
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


def evaluate_model(model, X_test, y_test) -> dict:
    """Evaluate a trained classifier on the test set.

    Parameters
    ----------
    model : sklearn estimator
        A fitted classifier.
    X_test : array-like
        Test feature matrix.
    y_test : array-like
        True labels.

    Returns
    -------
    dict
        Dictionary with accuracy, roc_auc, and classification_report.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "classification_report": classification_report(
            y_test, y_pred, target_names=["Low Risk", "High Risk"]
        ),
    }
    return metrics


def print_metrics(metrics: dict) -> None:
    """Pretty-print evaluation metrics."""
    print(f"\n{'='*50}")
    print(f"  Accuracy : {metrics['accuracy']:.4f}")
    print(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")
    print(f"\n  Classification Report:")
    print(metrics["classification_report"])
    print("=" * 50)


def plot_confusion_matrix(
    model, X_test, y_test, output_path: str = None
) -> None:
    """Plot and optionally save the confusion matrix.

    Parameters
    ----------
    model : sklearn estimator
        A fitted classifier.
    X_test : array-like
        Test feature matrix.
    y_test : array-like
        True labels.
    output_path : str, optional
        File path to save the figure (e.g. 'models/confusion_matrix.png').
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Low Risk", "High Risk"],
        yticklabels=["Low Risk", "High Risk"],
        ax=ax,
    )
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    ax.set_title("Confusion Matrix – Schisto Risk Classifier")
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150)
        print(f"Confusion matrix saved to {output_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
    output_path: str = None,
    top_n: int = 10,
) -> None:
    """Plot Random Forest feature importances.

    Parameters
    ----------
    feature_names : list[str]
        Names of the input features.
    importances : np.ndarray
        Feature importance values from the model.
    output_path : str, optional
        File path to save the figure.
    top_n : int
        Number of top features to display.
    """
    indices = np.argsort(importances)[::-1][:top_n]
    names = [feature_names[i] for i in indices]
    values = importances[indices]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(names[::-1], values[::-1], color="steelblue")
    ax.set_xlabel("Importance Score")
    ax.set_title(f"Top {top_n} Feature Importances – Random Forest")
    ax.bar_label(bars, fmt="%.3f", padding=3)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path, dpi=150)
        print(f"Feature importance plot saved to {output_path}")
    else:
        plt.show()
    plt.close(fig)
