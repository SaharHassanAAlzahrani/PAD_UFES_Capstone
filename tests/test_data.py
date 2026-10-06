from pad_multimodal.data import (
    MultimodalDataset,
    build_eval_transform,
    build_multimodal_loaders,
    build_train_transform,
)


def test_transforms_exist():
    train_transform = build_train_transform()
    eval_transform = build_eval_transform()

    assert train_transform is not None
    assert eval_transform is not None


def test_multimodal_dataset_rejects_length_mismatch():
    import pandas as pd
    import pytest

    dataframe = pd.DataFrame({
        "img_id": ["image_1.png", "image_2.png"],
        "patient_id": ["patient_1", "patient_2"],
        "diagnostic": ["ACK", "BCC"],
    })

    metadata = [
        [0.0, 1.0],
    ]

    with pytest.raises(ValueError):
        MultimodalDataset(
            dataframe=dataframe,
            metadata=metadata,
            image_lookup={},
            class_to_idx={"ACK": 0, "BCC": 1},
        )

def test_multimodal_loader_lengths(tmp_path):
    import numpy as np
    import pandas as pd
    from PIL import Image

    rows = []

    image_lookup = {}

    for index in range(4):
        img_id = f"image_{index}.png"
        image_path = tmp_path / img_id

        Image.new(
            "RGB",
            (32, 32),
        ).save(image_path)

        image_lookup[img_id] = image_path

        rows.append({
            "img_id": img_id,
            "patient_id": f"patient_{index}",
            "diagnostic": "ACK",
        })

    df = pd.DataFrame(rows)

    train_df = df.iloc[:3].copy()
    eval_df = df.iloc[3:].copy()

    train_metadata = np.zeros(
        (3, 33),
        dtype=np.float32,
    )

    eval_metadata = np.zeros(
        (1, 33),
        dtype=np.float32,
    )

    train_loader, eval_loader = build_multimodal_loaders(
        train_df=train_df,
        eval_df=eval_df,
        train_metadata=train_metadata,
        eval_metadata=eval_metadata,
        image_lookup=image_lookup,
        batch_size=2,
        num_workers=0,
    )

    assert len(train_loader.dataset) == 3
    assert len(eval_loader.dataset) == 1

    batch = next(iter(train_loader))

    assert batch["image"].shape[1:] == (3, 224, 224)
    assert batch["metadata"].shape[1] == 33