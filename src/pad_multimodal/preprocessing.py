import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from pad_multimodal.config import (
    CATEGORICAL_METADATA,
    NUMERIC_METADATA,
)


def build_metadata_preprocessor():
    """Create the fold-safe clinical metadata preprocessor."""

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                StandardScaler(),
                NUMERIC_METADATA,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                CATEGORICAL_METADATA,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

def fit_transform_metadata(
    train_df,
    eval_df,
):
    """Fit metadata preprocessing on training data only."""

    preprocessor = build_metadata_preprocessor()

    train_metadata = preprocessor.fit_transform(
        train_df
    )

    eval_metadata = preprocessor.transform(
        eval_df
    )

    train_metadata = np.asarray(
        train_metadata,
        dtype=np.float32,
    )

    eval_metadata = np.asarray(
        eval_metadata,
        dtype=np.float32,
    )

    if not np.isfinite(train_metadata).all():
        raise ValueError(
            "Non-finite values found in transformed training metadata."
        )

    if not np.isfinite(eval_metadata).all():
        raise ValueError(
            "Non-finite values found in transformed evaluation metadata."
        )

    return (
        train_metadata,
        eval_metadata,
        preprocessor,
    )