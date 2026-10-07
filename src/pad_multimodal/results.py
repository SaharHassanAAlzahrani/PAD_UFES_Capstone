from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    """Create and save a row-normalised confusion matrix plot."""
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=range(len(CLASS_NAMES)),
    )

    row_totals = matrix.sum(
        axis=1,
        keepdims=True,
    )

    matrix_percent = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(
            matrix,
            dtype=float,
        ),
        where=row_totals != 0,
    ) * 100

    figure, axis = plt.subplots(
        figsize=(8.5, 7)
    )

    image = axis.imshow(
        matrix_percent,
        cmap="Blues",
        vmin=0,
        vmax=100,
    )

    axis.set_xticks(
        range(len(CLASS_NAMES)),
        labels=CLASS_NAMES,
    )
    axis.set_yticks(
        range(len(CLASS_NAMES)),
        labels=CLASS_NAMES,
    )

    axis.set_xlabel(
        "Predicted Class",
        fontsize=11,
    )
    axis.set_ylabel(
        "True Class",
        fontsize=11,
    )

    display_names = {
        "image_only": "Image-Only",
        "concat": "Concatenation",
        "metablock_se": "MetaBlock-SE",
        "acrm": "ACRM",
    }

    display_name = display_names.get(
        model_name,
        model_name.replace("_", " ").title(),
    )

    axis.set_title(
        f"{display_name} Model: Confusion Matrix",
        fontsize=14,
        fontweight="normal",
        pad=14,
    )

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            percentage = matrix_percent[
                row,
                column,
            ]
            count = matrix[
                row,
                column,
            ]

            text_color = (
                "white"
                if percentage >= 50
                else "#1F2933"
            )

            axis.text(
                column,
                row,
                f"{percentage:.1f}%\n({count})",
                ha="center",
                va="center",
                fontsize=9,
                color=text_color,
            )

    colorbar = figure.colorbar(
        image,
        ax=axis,
        fraction=0.046,
        pad=0.04,
    )

    colorbar.set_label(
        "Percentage of True Class (%)",
        fontsize=10,
    )

    axis.tick_params(
        axis="both",
        labelsize=10,
    )

    for spine in axis.spines.values():
        spine.set_visible(False)

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
    """Create and save a publication-quality per-class recall plot."""
    recalls = recall_score(
        y_true,
        y_pred,
        labels=range(len(CLASS_NAMES)),
        average=None,
        zero_division=0,
    )

    recall_percent = recalls * 100

    figure, axis = plt.subplots(
        figsize=(9, 5.5)
    )

    positions = np.arange(
        len(CLASS_NAMES)
    )

    bars = axis.bar(
        positions,
        recall_percent,
        width=0.62,
        color="#234A73",
        edgecolor="#6A83AA",
        linewidth=1.4,
        zorder=3,
    )

    for bar, value in zip(
        bars,
        recall_percent,
    ):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.5,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#1F2933",
        )

    axis.set_xticks(
        positions,
        labels=CLASS_NAMES,
    )

    axis.set_ylim(0, 100)
    axis.set_yticks(
        range(0, 101, 20)
    )

    axis.set_xlabel(
        "Diagnostic Class",
        fontsize=11,
    )
    axis.set_ylabel(
        "Recall (%)",
        fontsize=11,
    )

    display_names = {
        "image_only": "Image-Only",
        "concat": "Concatenation",
        "metablock_se": "MetaBlock-SE",
        "acrm": "ACRM",
    }

    display_name = display_names.get(
        model_name,
        model_name.replace("_", " ").title(),
    )

    axis.set_title(
        f"{display_name} Model: Per-Class Recall",
        fontsize=14,
        fontweight="normal",
        pad=14,
    )

    axis.grid(
        axis="y",
        linestyle="--",
        linewidth=0.7,
        alpha=0.25,
        zorder=0,
    )

    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

    axis.spines["left"].set_linewidth(0.8)
    axis.spines["bottom"].set_linewidth(0.8)

    axis.tick_params(
        axis="both",
        labelsize=10,
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
