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


def load_existing_results(model_name):
    """Load previously completed folds for a model."""
    results_dir = Path("results") / model_name

    metrics_path = results_dir / "fold_metrics.csv"
    predictions_path = results_dir / "predictions.csv"

    if metrics_path.exists():
        fold_metrics = pd.read_csv(
            metrics_path
        ).to_dict("records")
    else:
        fold_metrics = []

    if predictions_path.exists():
        all_predictions = pd.read_csv(
            predictions_path
        ).to_dict("records")
    else:
        all_predictions = []

    return fold_metrics, all_predictions


def remove_existing_fold(
    fold_metrics,
    all_predictions,
    fold,
):
    """Remove an existing fold before replacing its results."""
    fold_metrics = [
        row
        for row in fold_metrics
        if int(row["fold"]) != fold
    ]

    all_predictions = [
        row
        for row in all_predictions
        if int(row["fold"]) != fold
    ]

    return fold_metrics, all_predictions


def save_complete_summary(
    model_name,
    fold_metrics,
    all_predictions,
):
    """Save summary tables and plots when all five folds exist."""
    metrics_df = pd.DataFrame(
        fold_metrics
    ).sort_values("fold")

    completed_folds = set(
        metrics_df["fold"].astype(int)
    )

    if completed_folds != set(range(5)):
        print(
            f"Completed folds: "
            f"{sorted(completed_folds)} / [0, 1, 2, 3, 4]"
        )
        return

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
        model_name=model_name,
        summary_metrics=summary_metrics,
    )

    predictions_df = pd.DataFrame(
        all_predictions
    ).sort_values(
        ["fold", "img_id"]
    )

    save_confusion_matrix_plot(
        model_name=model_name,
        y_true=predictions_df["target"],
        y_pred=predictions_df["prediction"],
    )

    save_per_class_recall_plot(
        model_name=model_name,
        y_true=predictions_df["target"],
        y_pred=predictions_df["prediction"],
    )

    print("All five folds completed.")
    print("Summary results saved.")
    print("Final plots saved.")


def main():
    """Run the requested experiment."""
    args = parse_args()

    metadata_path = Path(
        "metadata/pad_ufes_20_with_folds.csv"
    )
    image_dir = Path("data/images")

    dataframe = pd.read_csv(
        metadata_path
    )
    image_lookup = build_image_lookup(
        image_dir
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
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

    fold_metrics, all_predictions = (
        load_existing_results(
            args.model
        )
    )

    existing_folds = sorted(
        {
            int(row["fold"])
            for row in fold_metrics
        }
    )

    if existing_folds:
        print(
            f"Previously saved folds: "
            f"{existing_folds}"
        )

    for fold in folds:
        fold_metrics, all_predictions = (
            remove_existing_fold(
                fold_metrics=fold_metrics,
                all_predictions=all_predictions,
                fold=fold,
            )
        )

        print(
            f"\\nRunning outer fold {fold}"
        )

        result = run_outer_fold(
            dataframe=dataframe,
            image_lookup=image_lookup,
            model_name=args.model,
            outer_fold=fold,
            device=device,
        )

        metrics_row = {
            "fold": fold,
            "selected_epochs": result[
                "selected_epochs"
            ],
            "inner_best_macro_f1": result[
                "inner_best_macro_f1"
            ],
            "n_train": result[
                "n_outer_train"
            ],
            "n_test": result[
                "n_outer_test"
            ],
            **result["metrics"],
        }

        fold_metrics.append(
            metrics_row
        )

        prediction_result = result[
            "predictions"
        ]

        for index in range(
            len(
                prediction_result[
                    "targets"
                ]
            )
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

            for (
                class_index,
                class_name,
            ) in enumerate(
                CLASS_NAMES
            ):
                prediction_row[
                    f"prob_{class_name}"
                ] = prediction_result[
                    "probabilities"
                ][
                    index,
                    class_index,
                ]

            all_predictions.append(
                prediction_row
            )

        fold_metrics = sorted(
            fold_metrics,
            key=lambda row: int(
                row["fold"]
            ),
        )

        all_predictions = sorted(
            all_predictions,
            key=lambda row: (
                int(row["fold"]),
                str(row["img_id"]),
            ),
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

        save_complete_summary(
            model_name=args.model,
            fold_metrics=fold_metrics,
            all_predictions=all_predictions,
        )


if __name__ == "__main__":
    main()
