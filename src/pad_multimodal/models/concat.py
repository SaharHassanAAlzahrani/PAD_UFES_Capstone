import torch
import torch.nn as nn
from torchvision import models

from pad_multimodal.config import (
    DROPOUT,
    METADATA_HIDDEN_DIM,
    NUM_CLASSES,
)


class ConcatModel(nn.Module):
    """EfficientNet-B0 with conventional metadata concatenation."""

    def __init__(
        self,
        metadata_dim,
        num_classes=NUM_CLASSES,
        metadata_hidden_dim=METADATA_HIDDEN_DIM,
        dropout=DROPOUT,
    ):
        super().__init__()

        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        self.visual_encoder = models.efficientnet_b0(weights=weights)

        self.visual_dim = self.visual_encoder.classifier[1].in_features
        self.visual_encoder.classifier = nn.Identity()

        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_dim, metadata_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(metadata_hidden_dim, metadata_hidden_dim),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(
                self.visual_dim + metadata_hidden_dim,
                num_classes,
            ),
        )

    def forward(self, image, metadata):
        visual_features = self.visual_encoder(image)
        metadata_features = self.metadata_encoder(metadata)

        fused_features = torch.cat(
            [visual_features, metadata_features],
            dim=1,
        )

        return self.classifier(fused_features)