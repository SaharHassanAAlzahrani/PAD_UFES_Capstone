import pandas as pd
import numpy as np

from pad_multimodal.preprocessing import (
    build_metadata_preprocessor,
    fit_transform_metadata,
)

def test_metadata_preprocessor():
    df = pd.DataFrame({
        "age": [40, 55, 70],
        "region": ["ARM", "FACE", "BACK"],
        "itch": ["True", "False", "UNK"],
        "grew": ["False", "True", "False"],
        "hurt": ["False", "False", "True"],
        "changed": ["True", "False", "UNK"],
        "bleed": ["False", "True", "False"],
        "elevation": ["True", "False", "True"],
    })

    preprocessor = build_metadata_preprocessor()
    transformed = preprocessor.fit_transform(df)

    assert transformed.shape[0] == 3
    assert transformed.shape[1] > 1
    assert transformed.dtype.kind == "f"

def test_fit_transform_metadata():
    train_df = pd.DataFrame({
        "age": [40, 50, 60, 70],
        "region": ["ARM", "FACE", "BACK", "ARM"],
        "itch": ["True", "False", "UNK", "True"],
        "grew": ["False", "True", "False", "True"],
        "hurt": ["False", "False", "True", "False"],
        "changed": ["True", "False", "UNK", "False"],
        "bleed": ["False", "True", "False", "False"],
        "elevation": ["True", "False", "True", "False"],
    })

    eval_df = pd.DataFrame({
        "age": [55],
        "region": ["NOSE"],
        "itch": ["False"],
        "grew": ["True"],
        "hurt": ["False"],
        "changed": ["True"],
        "bleed": ["False"],
        "elevation": ["True"],
    })

    train_metadata, eval_metadata, preprocessor = (
        fit_transform_metadata(
            train_df,
            eval_df,
        )
    )

    assert train_metadata.shape[0] == 4
    assert eval_metadata.shape[0] == 1

    assert train_metadata.shape[1] == eval_metadata.shape[1]

    assert np.isfinite(train_metadata).all()
    assert np.isfinite(eval_metadata).all()