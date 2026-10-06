import pytest

from pad_multimodal.models.concat import ConcatModel
from pad_multimodal.models.factory import build_model
from pad_multimodal.models.image_only import ImageOnlyEfficientNet
from pad_multimodal.models.acrm import ACRM

def test_build_image_only_model():
    model = build_model(
        model_name="image_only",
    )

    assert isinstance(
        model,
        ImageOnlyEfficientNet,
    )


def test_build_concat_model():
    model = build_model(
        model_name="concat",
        metadata_dim=33,
    )

    assert isinstance(
        model,
        ConcatModel,
    )


def test_concat_requires_metadata_dim():
    with pytest.raises(ValueError):
        build_model(
            model_name="concat",
        )


def test_unknown_model_raises_error():
    with pytest.raises(ValueError):
        build_model(
            model_name="unknown",
        )

def test_build_acrm_model():
    model = build_model(
        model_name="acrm",
        metadata_dim=33,
    )

    assert isinstance(
        model,
        ACRM,
    )


def test_acrm_requires_metadata_dim():
    with pytest.raises(ValueError):
        build_model(
            model_name="acrm",
        )