import random

import numpy as np
import torch

from src.utils import get_device, set_seed


def test_set_seed_is_reproducible():
    set_seed(123)
    first = (random.random(), np.random.rand(), torch.rand(1).item())
    set_seed(123)
    second = (random.random(), np.random.rand(), torch.rand(1).item())
    assert first == second


def test_get_device_explicit_cpu():
    assert get_device("cpu") == torch.device("cpu")


def test_get_device_auto_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    assert get_device("auto") == torch.device("cpu")
