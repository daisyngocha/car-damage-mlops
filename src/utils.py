import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        if torch.accelerator.is_available():
            return torch.accelerator.current_accelerator()
        return torch.device("cpu")
    return torch.device(name)
