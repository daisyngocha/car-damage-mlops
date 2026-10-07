import torch
from omegaconf import OmegaConf

from src.model import build_model


def model_cfg(**kw):
    base = {"arch": "resnet18", "weights": None, "num_classes": 3, "replace_head": True}
    return OmegaConf.create({**base, **kw})


def test_build_model_replaces_head():
    model = build_model(model_cfg())
    assert model.fc.out_features == 3
    out = model.eval()(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 3)


def test_build_model_keeps_imagenet_head_when_not_replaced():
    model = build_model(model_cfg(replace_head=False))
    assert model.fc.out_features == 1000
