import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from pad_multimodal.train import (
    calculate_class_weights,
    predict,
    train_one_epoch,
)
class DummyDataset(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, index):
        return {
            "image": torch.randn(3, 16, 16),
            "metadata": torch.randn(4),
            "target": torch.tensor(index % 2),
            "img_id": f"image_{index}.png",
            "patient_id": f"patient_{index}",
        }


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.image_layer = nn.Linear(
            3 * 16 * 16,
            8,
        )

        self.metadata_layer = nn.Linear(
            4,
            8,
        )

        self.classifier = nn.Linear(
            16,
            2,
        )

    def forward(self, image, metadata=None):
        image = image.flatten(1)
        image = self.image_layer(image)

        metadata = self.metadata_layer(metadata)

        features = torch.cat(
            [image, metadata],
            dim=1,
        )

        return self.classifier(features)


def test_training_and_prediction():
    device = torch.device("cpu")

    model = DummyModel().to(device)

    loader = DataLoader(
        DummyDataset(),
        batch_size=4,
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    loss = train_one_epoch(
        model,
        loader,
        criterion,
        optimizer,
        device,
    )

    assert loss > 0

    results = predict(
        model,
        loader,
        criterion,
        device,
    )

    assert len(results["targets"]) == 8
    assert len(results["predictions"]) == 8
    assert results["probabilities"].shape == (8, 2)
    assert len(results["img_ids"]) == 8
    assert len(results["patient_ids"]) == 8

def test_class_weights():
    device = torch.device("cpu")

    targets = [0, 0, 0, 1, 1, 2]

    weights = calculate_class_weights(
        targets=targets,
        num_classes=3,
        device=device,
    )

    assert weights.shape == (3,)

    assert weights[2] > weights[1]
    assert weights[1] > weights[0]