import torch

from pad_multimodal.models.image_only import ImageOnlyEfficientNet
from pad_multimodal.models.concat import ConcatModel
from pad_multimodal.models.acrm import ACRM

def test_image_only_output_shape():
    model = ImageOnlyEfficientNet()

    images = torch.randn(2, 3, 224, 224)

    with torch.no_grad():
        output = model(images)

    assert output.shape == (2, 6)

def test_concat_output_shape():
    model = ConcatModel(metadata_dim=33)

    images = torch.randn(2, 3, 224, 224)
    metadata = torch.randn(2, 33)

    with torch.no_grad():
        output = model(images, metadata)

    assert output.shape == (2, 6)

def test_acrm_output_shape():
    model = ACRM(
        metadata_dim=33,
    )

    images = torch.randn(
        2,
        3,
        224,
        224,
    )

    metadata = torch.randn(
        2,
        33,
    )

    with torch.no_grad():
        output = model(
            images,
            metadata,
        )

    assert output.shape == (2, 6)

def test_acrm_metadata_pathway_receives_gradients():
    model = ACRM(
        metadata_dim=33,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    criterion = torch.nn.CrossEntropyLoss()

    images = torch.randn(
        4,
        3,
        224,
        224,
    )

    metadata = torch.randn(
        4,
        33,
    )

    targets = torch.tensor([
        0,
        1,
        2,
        3,
    ])

    model.train()

    optimizer.zero_grad()

    logits = model(
        images,
        metadata,
    )

    loss = criterion(
        logits,
        targets,
    )

    loss.backward()
    optimizer.step()

    optimizer.zero_grad()

    logits = model(
        images,
        metadata,
    )

    loss = criterion(
        logits,
        targets,
    )

    loss.backward()

    metadata_gradient = (
        model.metadata_encoder[0]
        .weight
        .grad
    )

    gate_gradient = (
        model.gate_head
        .weight
        .grad
    )

    gamma_gradient = (
        model.gamma_head
        .weight
        .grad
    )

    beta_gradient = (
        model.beta_head
        .weight
        .grad
    )

    assert metadata_gradient is not None
    assert gate_gradient is not None
    assert gamma_gradient is not None
    assert beta_gradient is not None

    assert metadata_gradient.abs().sum() > 0
    assert gate_gradient.abs().sum() > 0
    assert gamma_gradient.abs().sum() > 0
    assert beta_gradient.abs().sum() > 0

def test_metablock_se_forward():
    """MetaBlock-SE should return one logit per class."""
    from pad_multimodal.models.metablock_se import MetaBlockSEModel

    model = MetaBlockSEModel(
        metadata_dim=768,
        num_classes=6,
        pretrained=False,
    )

    model.eval()

    images = torch.randn(2, 3, 224, 224)
    metadata = torch.randn(2, 768)

    with torch.no_grad():
        logits = model(images, metadata)

    assert logits.shape == (2, 6)
    assert torch.isfinite(logits).all()

def test_factory_builds_metablock_se():
    """Factory should build a working MetaBlock-SE model."""
    from pad_multimodal.models.factory import build_model
    from pad_multimodal.models.metablock_se import MetaBlockSEModel

    model = build_model(
        model_name="metablock_se",
        metadata_dim=768,
    )

    assert isinstance(
        model,
        MetaBlockSEModel,
    )