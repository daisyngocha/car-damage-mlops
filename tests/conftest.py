"""Shared fixtures: a tiny CPU-only config and an on-disk ImageFolder dataset."""

from pathlib import Path

import pytest
import torch
from hydra import compose, initialize_config_dir
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

CONF_DIR = str(Path(__file__).resolve().parents[1] / "conf")
CLASSES = ["01-minor", "02-moderate", "03-severe"]


def make_image_folder(root: Path, n_per_class: int = 2, size: int = 40) -> None:
    for split in ("train", "val"):
        for i, cls in enumerate(CLASSES):
            d = root / split / cls
            d.mkdir(parents=True)
            for j in range(n_per_class):
                Image.new("RGB", (size, size), (80 * i, 40 * j, 120)).save(d / f"{j}.png")


@pytest.fixture
def image_root(tmp_path):
    root = tmp_path / "raw"
    make_image_folder(root)
    return root


@pytest.fixture
def cfg(tmp_path, image_root):
    """Real Hydra config from conf/, shrunk so the full pipeline runs on CPU in seconds."""
    overrides = [
        f"data.root={image_root}",
        f"data.processed_dir={image_root}",
        "data.img_size=32",
        "data.resize=36",
        "data.num_workers=0",
        "model.weights=null",
        "train.epochs=1",
        "train.batch_size=4",
        "device=cpu",
        f"paths.models_dir={tmp_path / 'models'}",
        f"paths.reports_dir={tmp_path / 'reports'}",
    ]
    with initialize_config_dir(config_dir=CONF_DIR, version_base=None):
        return compose(config_name="config", overrides=overrides)


@pytest.fixture
def tiny_loader():
    torch.manual_seed(0)
    images = torch.randn(8, 3, 8, 8)
    targets = torch.randint(0, 3, (8,))
    return DataLoader(TensorDataset(images, targets), batch_size=4)


def tiny_model(num_classes: int = 3) -> nn.Module:
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, num_classes))
