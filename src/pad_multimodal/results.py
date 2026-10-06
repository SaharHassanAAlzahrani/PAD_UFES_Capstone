from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
from sklearn.metrics import confusion_matrix, recall_score

from pad_multimodal.config import CLASS_NAMES

def get_model_results_dir(
    model_name,
    results_root="results",
):
    """Create and return the results directory for a model."""
    output_dir = Path(results_root) / model_name
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    return output_dir


def save_fold_metrics(
    model_name,
    fold_metrics,
    results_root="results",
):
    """Save one row of metrics for each outer fold."""
    output_dir = get_model_results_dir(
        model_name,
        results_root,
    )

    dataframe = pd.DataFrame(fold_metrics)

    output_path = output_dir / "fold_metrics.csv"
    dataframe.to_csv(
        output_path,
        index=False,
    )

    return output_path


def save_summary_metrics(
    model_name,
    summary_metrics,
    results_root="results",
):
    """Save report-ready summary metrics."""
    output_dir = get_model_results_dir(
        model_name,
        results_root,
    )

    dataframe = pd.DataFrame(summary_metrics)

    output_path = output_dir / "summary_metrics.csv"
    dataframe.to_csv(
        output_path,
        index=False,
    )

    return output_path


def save_predictions(
    model_name,
    predictions,
    results_root="results",
):
    """Save held-out predictions for later analysis."""
    output_dir = get_model_results_dir(
        model_name,
        results_root,
    )

    dataframe = pd.DataFrame(predictions)

    output_path = output_dir / "predictions.csv"
    dataframe.to_csv(
        output_path,
        index=False,
    )

    return output_path


def save_plot(
    model_name,
    filename,
    figure=None,
    results_root="results",
    dpi=300,
):
    """Save a publication-quality plot in the model results folder."""
    output_dir = get_model_results_dir(
        model_name,
        results_root,
    )

    output_path = output_dir / filename

    if figure is None:
        figure = plt.gcf()

    figure.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    return output_path

def save_confusion_matrix_plot(
    model_name,
    y_true,
    y_pred,
    results_root="results",
):
    """Create and save a confusion matrix plot."""
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=range(len(CLASS_NAMES)),
    )

    figure, axis = plt.subplots(
        figsize=(8, 7)
    )

    image = axis.imshow(matrix)

    axis.set_xticks(
        range(len(CLASS_NAMES)),
        labels=CLASS_NAMES,
    )
    axis.set_yticks(
        range(len(CLASS_NAMES)),
        labels=CLASS_NAMES,
    )

    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title(
        f"{model_name} confusion matrix"
    )

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
            )

    figure.colorbar(
        image,
        ax=axis,
    )
    figure.tight_layout()

    output_path = save_plot(
        model_name=model_name,
        filename="confusion_matrix.png",
        figure=figure,
        results_root=results_root,
    )

    plt.close(figure)

    return output_path


def save_per_class_recall_plot(
    model_name,
    y_true,
    y_pred,
    results_root="results",
):
    """Create and save per-class recall plot."""
    recalls = recall_score(
        y_true,
        y_pred,
        labels=range(len(CLASS_NAMES)),
        average=None,
        zero_division=0,
    )

    figure, axis = plt.subplots(
        figsize=(8, 5)
    )

    positions = np.arange(
        len(CLASS_NAMES)
    )

    axis.bar(
        positions,
        recalls,
    )

    axis.set_xticks(
        positions,
        labels=CLASS_NAMES,
    )
    axis.set_ylim(0, 1)
    axis.set_xlabel("Diagnostic class")
    axis.set_ylabel("Recall")
    axis.set_title(
        f"{model_name} per-class recall"
    )

    figure.tight_layout()

    output_path = save_plot(
        model_name=model_name,
        filename="per_class_recall.png",
        figure=figure,
        results_root=results_root,
    )

    plt.close(figure)

    return output_path