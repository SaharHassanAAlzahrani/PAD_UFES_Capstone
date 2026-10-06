import numpy as np
import torch


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):
    """Train a model for one epoch."""

    model.train()

    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["target"].to(device)

        metadata = batch.get("metadata")

        if metadata is not None:
            metadata = metadata.to(device)

        optimizer.zero_grad()

        logits = model(images, metadata)
        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / total_samples


@torch.no_grad()
def predict(
    model,
    loader,
    criterion,
    device,
):
    """Evaluate a model and return predictions and probabilities."""

    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_targets = []
    all_predictions = []
    all_probabilities = []
    all_img_ids = []
    all_patient_ids = []

    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["target"].to(device)

        metadata = batch.get("metadata")

        if metadata is not None:
            metadata = metadata.to(device)

        logits = model(images, metadata)
        loss = criterion(logits, targets)

        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

        batch_size = targets.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size

        all_targets.extend(
            targets.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy()
        )

        all_img_ids.extend(batch["img_id"])
        all_patient_ids.extend(batch["patient_id"])

    return {
        "loss": total_loss / total_samples,
        "targets": np.asarray(all_targets),
        "predictions": np.asarray(all_predictions),
        "probabilities": np.asarray(all_probabilities),
        "img_ids": all_img_ids,
        "patient_ids": all_patient_ids,
    }

def calculate_class_weights(
    targets,
    num_classes,
    device,
):
    """Calculate balanced class weights from the training partition only."""

    targets = np.asarray(targets)

    counts = np.bincount(
        targets,
        minlength=num_classes,
    )

    if np.any(counts == 0):
        raise ValueError(
            "Every class must be present in the training partition."
        )

    num_samples = len(targets)

    weights = num_samples / (
        num_classes * counts
    )

    return torch.tensor(
        weights,
        dtype=torch.float32,
        device=device,
    )