import torch.nn as nn
from torchvision import models

from pad_multimodal.config import NUM_CLASSES


class ImageOnlyEfficientNet(nn.Module):
    """EfficientNet-B0 image-only classification baseline."""

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        self.model = models.efficientnet_b0(weights=weights)

        in_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(
            in_features,
            num_classes,
        )

    def forward(self, image, metadata=None):
        return self.model(image)