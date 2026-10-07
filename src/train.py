import csv
import json
import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.optim.lr_scheduler import StepLR

from src.data import build_dataset, build_loader
from src.engine import predict, train_one_epoch
from src.model import build_model
from src.utils import get_device, set_seed

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> float:
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))
    set_seed(cfg.seed)
    device = get_device(cfg.device)

    # Thứ tự tạo model → loader giữ giống upstream main.py để cùng chuỗi random (parity test)
    model = build_model(cfg.model).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg.train.lr,
        momentum=cfg.train.momentum,
        weight_decay=cfg.train.weight_decay,
    )
    scheduler = StepLR(optimizer, step_size=cfg.train.lr_step_size, gamma=cfg.train.lr_gamma)

    train_ds = build_dataset(cfg.data, "train")
    val_ds = build_dataset(cfg.data, "val")
    if cfg.model.replace_head and len(train_ds.classes) != cfg.model.num_classes:
        raise ValueError(
            f"num_classes={cfg.model.num_classes} nhưng data có {len(train_ds.classes)} lớp"
        )
    train_loader = build_loader(train_ds, cfg.train.batch_size, cfg.data.num_workers, shuffle=True)
    val_loader = build_loader(val_ds, cfg.train.batch_size, cfg.data.num_workers, shuffle=False)
    log.info(
        "Train %d ảnh, val %d ảnh, lớp %s, device %s",
        len(train_ds),
        len(val_ds),
        train_ds.classes,
        device,
    )

    models_dir = Path(cfg.paths.models_dir)
    reports_dir = Path(cfg.paths.reports_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    history, best_acc = [], -1.0
    for epoch in range(1, cfg.train.epochs + 1):
        tr = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va = predict(model, val_loader, criterion, device)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step()
        history.append(
            {
                "epoch": epoch,
                "lr": lr,
                "train_loss": tr["loss"],
                "train_acc": tr["acc"],
                "val_loss": va["loss"],
                "val_acc": va["acc"],
            }
        )
        log.info(
            "Epoch %d/%d  train_loss %.4f  train_acc %.3f  val_loss %.4f  val_acc %.3f",
            epoch,
            cfg.train.epochs,
            tr["loss"],
            tr["acc"],
            va["loss"],
            va["acc"],
        )

        if va["acc"] > best_acc:
            best_acc = va["acc"]
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "classes": train_ds.classes,
                    "epoch": epoch,
                    "val_acc": best_acc,
                    "config": OmegaConf.to_container(cfg, resolve=True),
                },
                models_dir / "model.pt",
            )

    with open(reports_dir / "train_history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    (reports_dir / "train_summary.json").write_text(
        json.dumps({"best_val_acc": best_acc, "last_val_acc": history[-1]["val_acc"]}, indent=2)
    )
    log.info("Best val_acc %.3f → %s", best_acc, models_dir / "model.pt")
    return best_acc


if __name__ == "__main__":
    main()
