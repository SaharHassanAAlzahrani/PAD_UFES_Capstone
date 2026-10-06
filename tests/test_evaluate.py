import numpy as np

from pad_multimodal.evaluate import calculate_metrics


def test_calculate_metrics():
    y_true = np.array([0, 1, 2, 0, 1, 2])
    y_pred = np.array([0, 1, 2, 0, 1, 2])

    y_prob = np.array([
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9],
        [0.9, 0.05, 0.05],
        [0.05, 0.9, 0.05],
        [0.05, 0.05, 0.9],
    ])

    metrics = calculate_metrics(
        y_true,
        y_pred,
        y_prob,
    )

    assert metrics["macro_f1"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["macro_auroc"] == 1.0
    assert metrics["weighted_auroc"] == 1.0