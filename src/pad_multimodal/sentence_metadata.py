import pandas as pd

from pad_multimodal.config import SAFE_METADATA


FIELD_TO_LABEL = {
    "age": "Age",
    "region": "Lesion region",
    "itch": "Lesion itch",
    "grew": "Lesion grew",
    "hurt": "Lesion hurt",
    "changed": "Lesion changed",
    "bleed": "Lesion bled",
    "elevation": "Lesion elevation",
}


def build_metadata_sentence(row):
    """Convert leakage-safe clinical metadata into a clinical sentence."""
    parts = []

    for column in SAFE_METADATA:
        value = row[column]

        if pd.isna(value) or str(value).upper() == "UNK":
            continue

        parts.append(
            f"{FIELD_TO_LABEL[column]}: {str(value).lower()}"
        )

    if not parts:
        return "Patient History."

    return "Patient History: " + ", ".join(parts) + "."


def build_metadata_sentences(dataframe):
    """Create one clinical metadata sentence for each sample."""
    return dataframe.apply(
        build_metadata_sentence,
        axis=1,
    ).tolist()

def encode_metadata_sentences(
    dataframe,
    model,
):
    """Encode leakage-safe clinical sentences into fixed embeddings."""
    sentences = build_metadata_sentences(dataframe)

    embeddings = model.encode(
        sentences,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    return embeddings.astype("float32")