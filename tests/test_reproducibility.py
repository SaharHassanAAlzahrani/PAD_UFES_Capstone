import numpy as np
import torch

from pad_multimodal.reproducibility import set_seed


def test_set_seed_is_reproducible():
    set_seed(42)

    numpy_first = np.random.rand(3)
    torch_first = torch.rand(3)

    set_seed(42)

    numpy_second = np.random.rand(3)
    torch_second = torch.rand(3)

    assert np.allclose(numpy_first, numpy_second)
    assert torch.allclose(torch_first, torch_second)