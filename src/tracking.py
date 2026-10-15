"""W&B helpers: run có lineage code + data + config."""

import os
from pathlib import Path

import yaml
from omegaconf import DictConfig, OmegaConf

import wandb


def data_md5(dvc_file: str) -> str:
    """Lấy md5 từ file .dvc được truyền vào"""
    return yaml.safe_load(Path(dvc_file).read_text())["outs"][0]["md5"]


def init_run(cfg: DictConfig):
    config = OmegaConf.to_container(cfg, resolve=True)

    # Khai báo đường dẫn động 1 lần ở đây để dùng chung
    dvc_path = f"{cfg.data.raw_dir}.dvc"

    config["data_md5"] = data_md5(dvc_path)
    config["dvc_exp"] = os.getenv("DVC_EXP_NAME")  # DVC tự set khi chạy dvc exp run

    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        job_type="train",
        config=config,
    )
    run.define_metric("val/acc", summary="max")  # summary = best epoch, sweep tối ưu theo số này

    # Dataset artifact = chính con trỏ .dvc. Cùng md5 → W&B dedupe về cùng version.
    ds = wandb.Artifact("car-damage-data", type="dataset", metadata={"md5": config["data_md5"]})
    ds.add_file(dvc_path)  # add_file() → W&B lưu file .dvc vào artifact
    run.use_artifact(ds)  # khai báo input → lineage dataset → run
    return run


def log_model(run, path: str, cfg: DictConfig, metrics: dict) -> None:
    if not cfg.wandb.log_model:
        return
    art = wandb.Artifact(
        "car-damage-model",
        type="model",
        metadata={**metrics, "data_md5": run.config["data_md5"], "arch": cfg.model.arch},
    )
    art.add_file(path)
    run.log_artifact(art)  # khai báo output → lineage run → model
