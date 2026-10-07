import torch
from conftest import CLASSES
from PIL import Image

from src.data import build_dataset, build_loader, build_transforms


def test_val_transform_output_shape(cfg):
    tf = build_transforms(cfg.data, train=False)
    x = tf(Image.new("RGB", (50, 70)))
    assert x.shape == (3, cfg.data.img_size, cfg.data.img_size)


def test_train_transform_output_shape(cfg):
    tf = build_transforms(cfg.data, train=True)
    x = tf(Image.new("RGB", (50, 70)))
    assert x.shape == (3, cfg.data.img_size, cfg.data.img_size)


def test_build_dataset_reads_class_folders(cfg):
    train_ds = build_dataset(cfg.data, "train")
    val_ds = build_dataset(cfg.data, "val")
    assert train_ds.classes == CLASSES
    assert val_ds.classes == CLASSES
    assert len(train_ds) == len(val_ds) == 6


def test_build_loader_batches(cfg):
    loader = build_loader(
        build_dataset(cfg.data, "val"), batch_size=4, num_workers=0, shuffle=False
    )
    images, targets = next(iter(loader))
    assert images.shape == (4, 3, cfg.data.img_size, cfg.data.img_size)
    assert targets.dtype == torch.int64
