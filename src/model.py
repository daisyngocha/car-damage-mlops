from omegaconf import DictConfig
from torch import nn
from torchvision import models


def build_model(cfg: DictConfig) -> nn.Module:
    model = models.get_model(cfg.arch, weights=cfg.weights)
    if cfg.replace_head:
        model.fc = nn.Linear(model.fc.in_features, cfg.num_classes)
    return model
