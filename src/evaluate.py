import csv
import json
import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import classification_report, f1_score, recall_score
from torch import nn

from src.data import build_dataset, build_loader
from src.engine import predict
from src.model import build_model
from src.utils import get_device

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    device = get_device(cfg.device)
    ckpt = torch.load(Path(cfg.paths.models_dir) / "model.pt", map_location=device)

    # Dựng model theo config đã lưu trong checkpoint, không theo config hiện tại
    model_cfg = OmegaConf.create(ckpt["config"]["model"])
    if not model_cfg.replace_head:
        raise ValueError(
            "Checkpoint còn head 1000 lớp (parity run) → train lại với model.replace_head=true"
        )
    model_cfg.weights = None  # không tải weights ImageNet, nạp state_dict ngay sau
    model = build_model(model_cfg).to(device)
    model.load_state_dict(ckpt["state_dict"])

    val_ds = build_dataset(cfg.data, "val")
    if val_ds.classes != ckpt["classes"]:
        raise ValueError(f"Lớp của data {val_ds.classes} khác lớp lúc train {ckpt['classes']}")
    val_loader = build_loader(val_ds, cfg.train.batch_size, cfg.data.num_workers, shuffle=False)

    res = predict(model, val_loader, nn.CrossEntropyLoss(), device)

    labels = list(range(len(val_ds.classes)))
    sev = val_ds.classes.index("03-severe")
    metrics = {
        "accuracy": round(res["acc"], 3),
        "macro_f1": round(
            100 * f1_score(res["y_true"], res["y_pred"], labels=labels, average="macro"),
            3,
        ),
        "recall_severe": round(
            100 * recall_score(res["y_true"], res["y_pred"], labels=[sev], average=None)[0], 3
        ),
        "val_loss": round(res["loss"], 4),
        "n_val": len(val_ds),
        "checkpoint_epoch": ckpt["epoch"],
    }

    reports_dir = Path(cfg.paths.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    # mỗi dòng 1 ảnh val, dùng cho DVC plots (confusion matrix) ở G6
    with open(reports_dir / "confusion.csv", "w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["actual", "predicted"])
        writer.writerows(
            (val_ds.classes[t], val_ds.classes[p]) for t, p in zip(res["y_true"], res["y_pred"])
        )

    log.info("Metrics: %s", metrics)
    log.info(
        "\n%s",
        classification_report(
            res["y_true"],
            res["y_pred"],
            labels=labels,
            target_names=val_ds.classes,
            digits=3,
            zero_division=0,
        ),
    )


if __name__ == "__main__":
    main()
