import argparse
from pathlib import Path

import pandas as pd
import torch

from pad_multimodal.config import CLASS_NAMES
from pad_multimodal.data import build_image_lookup
from pad_multimodal.experiment import run_outer_fold
from pad_multimodal.results import (
    save_confusion_matrix_plot,
    save_fold_metrics,
    save_per_class_recall_plot,
    save_predictions,
    save_summary_metrics,
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a PAD-UFES-20 model experiment."
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "image_only",
            "concat",
            "metablock_se",
            "acrm",
        ],
    )

    parser.add_argument(
        "--fold",
        type=int,
        choices=[0, 1, 2, 3, 4],
        default=None,
        help="Run one fold only. Omit to run all five folds.",
    )

    return parser.parse_args()


def main():
    """Run the requested experiment."""

    args = parse_args()

    metadata_path = Path(
        "metadata/pad_ufes_20_with_folds.csv"
    )
    image_dir = Path("data/images")

    dataframe = pd.read_csv(metadata_path)
    image_lookup = build_image_lookup(image_dir)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    folds = (
        [args.fold]
        if args.fold is not None
        else list(range(5))
    )

    print(f"Model: {args.model}")
    print(f"Device: {device}")
    print(f"Folds: {folds}")
    print(f"Samples: {len(dataframe)}")
    print(f"Images found: {len(image_lookup)}")

    fold_metrics = []
    all_predictions = []

    for fold in folds:
        print(f"\nRunning outer fold {fold}")

        result = run_outer_fold(
            dataframe=dataframe,
            image_lookup=image_lookup,
            model_name=args.model,
            outer_fold=fold,
            device=device,
        )

        metrics_row = {
            "fold": fold,
            "selected_epochs": result["selected_epochs"],
            "inner_best_macro_f1": result[
                "inner_best_macro_f1"
            ],
            "n_train": result["n_outer_train"],
            "n_test": result["n_outer_test"],
            **result["metrics"],
        }

        fold_metrics.append(
            metrics_row
        )

        prediction_result = result[
            "predictions"
        ]

        for index in range(
            len(prediction_result["targets"])
        ):
            prediction_row = {
                "fold": fold,
                "img_id": prediction_result[
                    "img_ids"
                ][index],
                "patient_id": prediction_result[
                    "patient_ids"
                ][index],
                "target": prediction_result[
                    "targets"
                ][index],
                "prediction": prediction_result[
                    "predictions"
                ][index],
            }

            for class_index, class_name in enumerate(
                CLASS_NAMES
            ):
                prediction_row[
                    f"prob_{class_name}"
                ] = prediction_result[
                    "probabilities"
                ][index, class_index]

            all_predictions.append(
                prediction_row
            )

        save_fold_metrics(
            model_name=args.model,
            fold_metrics=fold_metrics,
        )

        save_predictions(
            model_name=args.model,
            predictions=all_predictions,
        )

        print(
            f"Fold {fold} completed:",
            result["metrics"],
        )

        print(
            f"Fold {fold} results saved."
        )

    if len(fold_metrics) > 1:
        metrics_df = pd.DataFrame(
            fold_metrics
        )

        metric_names = [
            "macro_f1",
            "balanced_accuracy",
            "macro_auroc",
            "weighted_auroc",
            "loss",
        ]

        summary_metrics = []

        for metric_name in metric_names:
            summary_metrics.append(
                {
                    "metric": metric_name,
                    "mean": metrics_df[
                        metric_name
                    ].mean(),
                    "std": metrics_df[
                        metric_name
                    ].std(),
                }
            )

        save_summary_metrics(
            model_name=args.model,
            summary_metrics=summary_metrics,
        )

        predictions_df = pd.DataFrame(
            all_predictions
        )

        save_confusion_matrix_plot(
            model_name=args.model,
            y_true=predictions_df["target"],
            y_pred=predictions_df["prediction"],
        )

        save_per_class_recall_plot(
            model_name=args.model,
            y_true=predictions_df["target"],
            y_pred=predictions_df["prediction"],
        )

        print(
            "\nSummary results saved."
        )

        print(
            "Final plots saved."
        )


if __name__ == "__main__":
    main()