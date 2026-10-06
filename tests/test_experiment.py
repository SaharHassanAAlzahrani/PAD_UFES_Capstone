import pandas as pd
import torch
import torch.nn as nn
from unittest.mock import patch
import numpy as np
from torch.utils.data import DataLoader, Dataset

from pad_multimodal.experiment import (
    create_inner_split,
    create_outer_split,
    evaluate_final_model,
    select_training_epochs,
    train_final_model,
    run_outer_fold,
)


def test_inner_split_has_no_patient_overlap():
    rows = []

    classes = ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"]

    for class_name in classes:
        for patient_number in range(10):
            patient_id = f"{class_name}_{patient_number}"

            rows.append({
                "patient_id": patient_id,
                "diagnostic": class_name,
            })

            rows.append({
                "patient_id": patient_id,
                "diagnostic": class_name,
            })

    df = pd.DataFrame(rows)

    train_df, val_df = create_inner_split(df)

    train_patients = set(train_df["patient_id"])
    val_patients = set(val_df["patient_id"])

    assert len(train_df) + len(val_df) == len(df)
    assert train_patients.isdisjoint(val_patients)

    assert set(train_df["diagnostic"]) == set(classes)
    assert set(val_df["diagnostic"]) == set(classes)


def test_outer_split_uses_fixed_folds():
    df = pd.DataFrame({
        "patient_id": [
            "p1", "p1",
            "p2", "p2",
            "p3", "p3",
        ],
        "diagnostic": [
            "ACK", "ACK",
            "BCC", "BCC",
            "MEL", "MEL",
        ],
        "fold": [
            0, 0,
            1, 1,
            2, 2,
        ],
    })

    train_df, test_df = create_outer_split(
        dataframe=df,
        outer_fold=1,
    )

    assert set(test_df["patient_id"]) == {"p2"}
    assert set(train_df["patient_id"]) == {"p1", "p3"}

    assert set(train_df["patient_id"]).isdisjoint(
        set(test_df["patient_id"])
    )


class DummyTrainingDataset(Dataset):
    def __len__(self):
        return 12

    def __getitem__(self, index):
        return {
            "image": torch.randn(3, 8, 8),
            "target": torch.tensor(index % 6),
            "img_id": f"image_{index}.png",
            "patient_id": f"patient_{index}",
        }


class DummyTrainingModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.classifier = nn.Linear(
            3 * 8 * 8,
            6,
        )

    def forward(self, image, metadata=None):
        return self.classifier(
            image.flatten(1)
        )


def test_select_training_epochs():
    rows = []

    classes = [
        "ACK",
        "BCC",
        "MEL",
        "NEV",
        "SCC",
        "SEK",
    ]

    for index in range(12):
        rows.append({
            "diagnostic": classes[index % 6],
        })

    train_df = pd.DataFrame(rows)

    loader = DataLoader(
        DummyTrainingDataset(),
        batch_size=6,
    )

    model = DummyTrainingModel()

    result = select_training_epochs(
        model=model,
        train_loader=loader,
        val_loader=loader,
        train_df=train_df,
        device=torch.device("cpu"),
        max_epochs=2,
        patience=2,
    )

    assert 1 <= result["selected_epochs"] <= 2

    assert len(result["history"]) >= 1

    assert (
        len(result["learning_rate_schedule"])
        == result["selected_epochs"]
    )

def test_train_final_model():
    rows = []

    classes = [
        "ACK",
        "BCC",
        "MEL",
        "NEV",
        "SCC",
        "SEK",
    ]

    for index in range(12):
        rows.append({
            "diagnostic": classes[index % 6],
        })

    train_df = pd.DataFrame(rows)

    loader = DataLoader(
        DummyTrainingDataset(),
        batch_size=6,
    )

    model = DummyTrainingModel()

    result = train_final_model(
        model=model,
        train_loader=loader,
        train_df=train_df,
        device=torch.device("cpu"),
        selected_epochs=2,
        learning_rate_schedule=[
            1e-4,
            5e-5,
        ],
    )

    assert result["model"] is model

    assert len(result["history"]) == 2

    assert (
        result["history"][0]["learning_rate"]
        == 1e-4
    )

    assert (
        result["history"][1]["learning_rate"]
        == 5e-5
    )

def test_evaluate_final_model():
    loader = DataLoader(
        DummyTrainingDataset(),
        batch_size=6,
    )

    model = DummyTrainingModel()

    criterion = nn.CrossEntropyLoss()

    result = evaluate_final_model(
        model=model,
        test_loader=loader,
        criterion=criterion,
        device=torch.device("cpu"),
    )

    assert "metrics" in result
    assert "predictions" in result

    assert "macro_f1" in result["metrics"]
    assert "balanced_accuracy" in result["metrics"]
    assert "macro_auroc" in result["metrics"]
    assert "weighted_auroc" in result["metrics"]
    assert "loss" in result["metrics"]

    predictions = result["predictions"]

    assert len(predictions["targets"]) == 12
    assert len(predictions["predictions"]) == 12
    assert predictions["probabilities"].shape == (12, 6)
    assert len(predictions["img_ids"]) == 12
    assert len(predictions["patient_ids"]) == 12


def test_run_outer_fold_orchestration():
    classes = [
        "ACK",
        "BCC",
        "MEL",
        "NEV",
        "SCC",
        "SEK",
    ]

    rows = []

    for fold in range(5):
        for class_name in classes:
            for patient_number in range(5):
                rows.append({
                    "patient_id": (
                        f"fold{fold}_{class_name}_{patient_number}"
                    ),
                    "diagnostic": class_name,
                    "fold": fold,
                    "img_id": (
                        f"fold{fold}_{class_name}_{patient_number}.png"
                    ),
                })

    dataframe = pd.DataFrame(rows)

    dummy_metadata = np.zeros(
        (120, 33),
        dtype=np.float32,
    )

    selection_result = {
        "selected_epochs": 2,
        "best_macro_f1": 0.5,
        "learning_rate_schedule": [
            1e-4,
            1e-4,
        ],
        "history": [],
    }

    final_training_result = {
        "model": object(),
        "criterion": object(),
        "history": [],
    }

    evaluation_result = {
        "metrics": {
            "macro_f1": 0.6,
            "balanced_accuracy": 0.6,
            "macro_auroc": 0.8,
            "weighted_auroc": 0.8,
            "loss": 1.0,
        },
        "predictions": {
            "targets": np.zeros(30),
            "predictions": np.zeros(30),
            "probabilities": np.zeros((30, 6)),
            "img_ids": [],
            "patient_ids": [],
        },
    }

    with (
        patch(
            "pad_multimodal.experiment.fit_transform_metadata",
            return_value=(
                dummy_metadata,
                dummy_metadata,
                object(),
            ),
        ),
        patch(
            "pad_multimodal.experiment.build_multimodal_loaders",
            return_value=(object(), object()),
        ),
        patch(
            "pad_multimodal.experiment.build_model",
            return_value=object(),
        ),
        patch(
            "pad_multimodal.experiment.select_training_epochs",
            return_value=selection_result,
        ),
        patch(
            "pad_multimodal.experiment.train_final_model",
            return_value=final_training_result,
        ),
        patch(
            "pad_multimodal.experiment.evaluate_final_model",
            return_value=evaluation_result,
        ),
    ):
        result = run_outer_fold(
            dataframe=dataframe,
            image_lookup={},
            model_name="concat",
            outer_fold=0,
            device=torch.device("cpu"),
        )

    assert result["model_name"] == "concat"
    assert result["outer_fold"] == 0
    assert result["selected_epochs"] == 2
    assert result["metadata_dim"] == 33

    assert result["n_outer_train"] == 120
    assert result["n_outer_test"] == 30

    assert result["metrics"]["macro_f1"] == 0.6