import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


class MMFNetModel(nn.Module):
    """
    Adapted reimplementation of MMF-Net for the common project protocol.

    The model follows the main MMF-Net fusion design:
    image and metadata encoding, intra-modality self-attention,
    bidirectional inter-modality cross-attention, feature concatenation,
    and final classification.

    EfficientNet-B0 is used as the common visual backbone so that the
    benchmark differs from the other project models primarily in its
    multimodal fusion strategy.
    """

    def __init__(
        self,
        metadata_dim,
        num_classes=6,
        embed_dim=128,
        num_heads=4,
        dropout=0.2,
        pretrained=True,
    ):
        super().__init__()

        if metadata_dim is None:
            raise ValueError(
                "metadata_dim is required for MMF-Net."
            )

        if embed_dim % num_heads != 0:
            raise ValueError(
                "embed_dim must be divisible by num_heads."
            )
        
        weights = (
            EfficientNet_B0_Weights.DEFAULT
            if pretrained
            else None
        )
        backbone = efficientnet_b0(
            weights=weights
        )

        image_feature_dim = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()

        self.image_encoder = backbone

        self.image_projection = nn.Sequential(
            nn.Linear(image_feature_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.metadata_encoder = nn.Sequential(
            nn.Linear(metadata_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )

        self.image_self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.metadata_self_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.image_cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.metadata_cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.image_norm = nn.LayerNorm(embed_dim)
        self.metadata_norm = nn.LayerNorm(embed_dim)

        self.image_cross_norm = nn.LayerNorm(embed_dim)
        self.metadata_cross_norm = nn.LayerNorm(embed_dim)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, image, metadata):
        image_features = self.image_encoder(image)
        image_features = self.image_projection(image_features)

        metadata_features = self.metadata_encoder(metadata)

        image_tokens = image_features.unsqueeze(1)
        metadata_tokens = metadata_features.unsqueeze(1)

        image_self, _ = self.image_self_attention(
            image_tokens,
            image_tokens,
            image_tokens,
            need_weights=False,
        )

        metadata_self, _ = self.metadata_self_attention(
            metadata_tokens,
            metadata_tokens,
            metadata_tokens,
            need_weights=False,
        )

        image_self = self.image_norm(
            image_tokens + image_self
        )

        metadata_self = self.metadata_norm(
            metadata_tokens + metadata_self
        )

        image_cross, _ = self.image_cross_attention(
            query=image_self,
            key=metadata_self,
            value=metadata_self,
            need_weights=False,
        )

        metadata_cross, _ = self.metadata_cross_attention(
            query=metadata_self,
            key=image_self,
            value=image_self,
            need_weights=False,
        )

        image_cross = self.image_cross_norm(
            image_self + image_cross
        )

        metadata_cross = self.metadata_cross_norm(
            metadata_self + metadata_cross
        )

        image_final = image_cross.squeeze(1)
        metadata_final = metadata_cross.squeeze(1)

        fused_features = torch.cat(
            [image_final, metadata_final],
            dim=1,
        )

        return self.classifier(fused_features)