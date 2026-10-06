from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

from pad_multimodal.config import (
    BATCH_SIZE,
    CLASS_TO_IDX,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
)


def build_train_transform():
    """Image transformations used during training."""

    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.02,
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])


def build_eval_transform():
    """Deterministic image transformations for validation and testing."""

    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])


def build_image_lookup(image_dir):
    """Map image filenames to their full paths."""

    image_dir = Path(image_dir)

    return {
        path.name: path
        for path in image_dir.rglob("*.png")
    }


class ImageDataset(Dataset):
    """PAD-UFES-20 image-only dataset."""

    def __init__(
        self,
        dataframe,
        image_lookup,
        class_to_idx,
        transform=None,
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_lookup = image_lookup
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        img_id = row["img_id"]
        image_path = self.image_lookup[img_id]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        target = self.class_to_idx[row["diagnostic"]]

        return {
            "image": image,
            "target": torch.tensor(target, dtype=torch.long),
            "img_id": img_id,
            "patient_id": row["patient_id"],
        }


class MultimodalDataset(Dataset):
    """PAD-UFES-20 image and clinical metadata dataset."""

    def __init__(
        self,
        dataframe,
        metadata,
        image_lookup,
        class_to_idx,
        transform=None,
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.metadata = torch.as_tensor(
            metadata,
            dtype=torch.float32,
        )
        self.image_lookup = image_lookup
        self.class_to_idx = class_to_idx
        self.transform = transform

        if len(self.dataframe) != len(self.metadata):
            raise ValueError(
                "Dataframe and metadata must contain "
                "the same number of samples."
            )

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]

        img_id = row["img_id"]
        image_path = self.image_lookup[img_id]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        target = self.class_to_idx[row["diagnostic"]]

        return {
            "image": image,
            "metadata": self.metadata[index],
            "target": torch.tensor(target, dtype=torch.long),
            "img_id": img_id,
            "patient_id": row["patient_id"],
        }
    
def build_multimodal_loaders(
    train_df,
    eval_df,
    train_metadata,
    eval_metadata,
    image_lookup,
    batch_size=BATCH_SIZE,
    num_workers=2,
):
    """Build training and evaluation loaders for multimodal models."""

    train_dataset = MultimodalDataset(
        dataframe=train_df,
        metadata=train_metadata,
        image_lookup=image_lookup,
        class_to_idx=CLASS_TO_IDX,
        transform=build_train_transform(),
    )

    eval_dataset = MultimodalDataset(
        dataframe=eval_df,
        metadata=eval_metadata,
        image_lookup=image_lookup,
        class_to_idx=CLASS_TO_IDX,
        transform=build_eval_transform(),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, eval_loader