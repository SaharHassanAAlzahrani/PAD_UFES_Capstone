import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_fold_seed(base_seed: int, fold: int) -> int:
    """Set a reproducible seed for a specific outer fold."""

    fold_seed = base_seed + fold
    set_seed(fold_seed)

    return fold_seed