from pathlib import Path

# Reproducibility
SEED = 42

# Dataset
CLASS_NAMES = [
    "ACK",
    "BCC",
    "MEL",
    "NEV",
    "SCC",
    "SEK",
]

CLASS_TO_IDX = {
    label: idx
    for idx, label in enumerate(CLASS_NAMES)
}

NUM_CLASSES = len(CLASS_NAMES)

# Metadata
SAFE_METADATA = [
    "age",
    "region",
    "itch",
    "grew",
    "hurt",
    "changed",
    "bleed",
    "elevation",
]

NUMERIC_METADATA = [
    "age",
]

CATEGORICAL_METADATA = [
    "region",
    "itch",
    "grew",
    "hurt",
    "changed",
    "bleed",
    "elevation",
]

# Images
IMAGE_SIZE = 224

IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406,
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225,
]

# Training
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
MAX_EPOCHS = 25
EARLY_STOPPING_PATIENCE = 5

# Cross-validation
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 5

# Model
BACKBONE = "efficientnet_b0"
PRETRAINED = True

METADATA_HIDDEN_DIM = 128
DROPOUT = 0.2