import numpy as np

from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


def calculate_metrics(
    y_true,
    y_pred,
    y_prob=None,
):
    """Calculate the common classification metrics for all models."""

    metrics = {
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            y_pred,
        ),
    }

    if y_prob is not None:
        y_true = np.asarray(y_true)
        y_prob = np.asarray(y_prob)

        metrics["macro_auroc"] = roc_auc_score(
            y_true,
            y_prob,
            multi_class="ovr",
            average="macro",
        )

        metrics["weighted_auroc"] = roc_auc_score(
            y_true,
            y_prob,
            multi_class="ovr",
            average="weighted",
        )

    return metrics