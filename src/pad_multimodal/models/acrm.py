import torch
import torch.nn as nn

from torchvision import models

from pad_multimodal.config import (
    DROPOUT,
    METADATA_HIDDEN_DIM,
    NUM_CLASSES,
)


class ACRM(nn.Module):
    """Adaptive Clinical Residual Modulation model."""

    def __init__(
        self,
        metadata_dim,
        num_classes=NUM_CLASSES,
        metadata_hidden_dim=METADATA_HIDDEN_DIM,
        dropout=DROPOUT,
    ):
        super().__init__()

        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1

        self.visual_encoder = models.efficientnet_b0(
            weights=weights,
        )

        self.visual_dim = (
            self.visual_encoder.classifier[1].in_features
        )

        self.visual_encoder.classifier = nn.Identity()

        self.metadata_encoder = nn.Sequential(
            nn.Linear(
                metadata_dim,
                metadata_hidden_dim,
            ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(
                metadata_hidden_dim,
                metadata_hidden_dim,
            ),
            nn.ReLU(),
        )

        self.gamma_head = nn.Linear(
            metadata_hidden_dim,
            self.visual_dim,
        )

        self.beta_head = nn.Linear(
            metadata_hidden_dim,
            self.visual_dim,
        )

        self.gate_head = nn.Linear(
            metadata_hidden_dim,
            self.visual_dim,
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(
                self.visual_dim,
                num_classes,
            ),
        )

        nn.init.zeros_(
            self.gamma_head.weight
        )
        nn.init.zeros_(
            self.gamma_head.bias
        )

        nn.init.zeros_(
            self.beta_head.weight
        )
        nn.init.zeros_(
            self.beta_head.bias
        )

    def forward(
        self,
        image,
        metadata,
    ):
        visual_features = self.visual_encoder(
            image
        )

        metadata_features = self.metadata_encoder(
            metadata
        )

        gamma = self.gamma_head(
            metadata_features
        )

        beta = self.beta_head(
            metadata_features
        )

        gate = torch.sigmoid(
            self.gate_head(
                metadata_features
            )
        )

        modulation = (
            gamma * visual_features
            + beta
        )

        modulated_features = (
            visual_features
            + gate * modulation
        )

        return self.classifier(
            modulated_features
        )