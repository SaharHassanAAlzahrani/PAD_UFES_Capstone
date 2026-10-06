import torch
from torch import nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    efficientnet_b0,
)


class MetaBlock(nn.Module):
    """Metadata-guided visual feature modulation."""

    def __init__(
        self,
        visual_channels,
        metadata_dim,
    ):
        super().__init__()

        self.fb = nn.Sequential(
            nn.Linear(metadata_dim, visual_channels),
            nn.BatchNorm1d(visual_channels),
        )

        self.gb = nn.Sequential(
            nn.Linear(metadata_dim, visual_channels),
            nn.BatchNorm1d(visual_channels),
        )

    def forward(
        self,
        visual_features,
        metadata,
    ):
        t1 = self.fb(metadata)
        t2 = self.gb(metadata)

        return torch.sigmoid(
            torch.tanh(
                visual_features * t1.unsqueeze(-1)
            )
            + t2.unsqueeze(-1)
        )


class MetaBlockSEModel(nn.Module):
    """EfficientNet-B0 with sentence-embedded MetaBlock fusion."""

    def __init__(
        self,
        metadata_dim=768,
        num_classes=6,
        pretrained=True,
    ):
        super().__init__()

        weights = (
            EfficientNet_B0_Weights.DEFAULT
            if pretrained
            else None
        )

        backbone = efficientnet_b0(weights=weights)

        self.features = backbone.features
        self.avgpool = backbone.avgpool

        self.visual_channels = 1280
        self.metablock_channels = 32

        if self.visual_channels % self.metablock_channels != 0:
            raise ValueError(
                "Visual feature dimension must be divisible "
                "by MetaBlock channels."
            )

        self.metablock = MetaBlock(
            visual_channels=self.metablock_channels,
            metadata_dim=metadata_dim,
        )

        self.classifier = nn.Linear(
            self.visual_channels,
            num_classes,
        )

    def forward(
        self,
        image,
        metadata,
    ):
        if metadata is None:
            raise ValueError(
                "MetaBlock-SE requires sentence embeddings."
            )

        features = self.features(image)
        features = self.avgpool(features)
        features = torch.flatten(features, 1)

        features = features.view(
            features.size(0),
            self.metablock_channels,
            -1,
        )

        features = self.metablock(
            features,
            metadata.float(),
        )

        features = torch.flatten(features, 1)

        return self.classifier(features)