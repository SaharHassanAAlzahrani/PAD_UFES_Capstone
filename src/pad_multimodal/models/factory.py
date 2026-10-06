from pad_multimodal.models.acrm import ACRM
from pad_multimodal.models.concat import ConcatModel
from pad_multimodal.models.image_only import ImageOnlyEfficientNet
from pad_multimodal.models.metablock_se import MetaBlockSEModel


def build_model(
    model_name,
    metadata_dim=None,
):
    """Create a fresh model instance."""

    if model_name == "image_only":
        return ImageOnlyEfficientNet()

    if model_name == "concat":
        if metadata_dim is None:
            raise ValueError(
                "metadata_dim is required for the concat model."
            )

        return ConcatModel(
            metadata_dim=metadata_dim,
        )

    if model_name == "acrm":
        if metadata_dim is None:
            raise ValueError(
                "metadata_dim is required for ACRM."
            )

        return ACRM(
            metadata_dim=metadata_dim,
        )

    if model_name == "metablock_se":
        if metadata_dim is None:
            raise ValueError(
                "metadata_dim is required for MetaBlock-SE."
            )

        return MetaBlockSEModel(
            metadata_dim=metadata_dim,
        )

    raise ValueError(
        f"Unknown model: {model_name}"
    )