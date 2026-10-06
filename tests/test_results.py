import matplotlib.pyplot as plt
import pandas as pd

from pad_multimodal.results import (
    save_fold_metrics,
    save_plot,
    save_predictions,
    save_summary_metrics,
)


def test_result_files_are_saved(tmp_path):
    """Result tables and plots should be saved correctly."""
    model_name = "test_model"

    fold_path = save_fold_metrics(
        model_name=model_name,
        fold_metrics=[
            {
                "fold": 0,
                "macro_f1": 0.60,
                "balanced_accuracy": 0.62,
            },
            {
                "fold": 1,
                "macro_f1": 0.64,
                "balanced_accuracy": 0.65,
            },
        ],
        results_root=tmp_path,
    )

    summary_path = save_summary_metrics(
        model_name=model_name,
        summary_metrics=[
            {
                "metric": "macro_f1",
                "mean": 0.62,
                "std": 0.02,
            }
        ],
        results_root=tmp_path,
    )

    prediction_path = save_predictions(
        model_name=model_name,
        predictions=[
            {
                "img_id": "sample.png",
                "target": 0,
                "prediction": 1,
            }
        ],
        results_root=tmp_path,
    )

    figure, axis = plt.subplots()
    axis.plot([0, 1], [0, 1])

    plot_path = save_plot(
        model_name=model_name,
        filename="test_plot.png",
        figure=figure,
        results_root=tmp_path,
    )

    plt.close(figure)

    assert fold_path.exists()
    assert summary_path.exists()
    assert prediction_path.exists()
    assert plot_path.exists()

    fold_df = pd.read_csv(fold_path)
    summary_df = pd.read_csv(summary_path)
    prediction_df = pd.read_csv(prediction_path)

    assert len(fold_df) == 2
    assert len(summary_df) == 1
    assert len(prediction_df) == 1