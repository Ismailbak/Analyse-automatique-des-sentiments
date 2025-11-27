"""Evaluation helpers: confusion matrix plots and metrics."""
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


def plot_confusion(y_true, y_pred, labels=None, figsize=(6,6), cmap='Blues'):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    if labels:
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
    return fig


def classification_summary(y_true, y_pred):
    return classification_report(y_true, y_pred)
