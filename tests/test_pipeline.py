"""End-to-end smoke tests: train.main → evaluate.main on a tiny on-disk dataset."""

import json
from pathlib import Path

import pytest
import torch
from conftest import CLASSES
from omegaconf import open_dict

from src import evaluate, train


def test_train_writes_checkpoint_and_reports(cfg):
    best_acc = train.main(cfg)

    models_dir, reports_dir = Path(cfg.paths.models_dir), Path(cfg.paths.reports_dir)
    ckpt = torch.load(models_dir / "model.pt", weights_only=False)
    assert ckpt["classes"] == CLASSES
    assert ckpt["epoch"] == 1
    assert (reports_dir / "train_history.csv").exists()
    summary = json.loads((reports_dir / "train_summary.json").read_text())
    assert summary["best_val_acc"] == pytest.approx(best_acc)


def test_train_rejects_wrong_num_classes(cfg):
    with open_dict(cfg):
        cfg.model.num_classes = 5
    with pytest.raises(ValueError, match="num_classes"):
        train.main(cfg)


def test_evaluate_writes_metrics_and_confusion(cfg):
    train.main(cfg)
    evaluate.main(cfg)

    reports_dir = Path(cfg.paths.reports_dir)
    metrics = json.loads((reports_dir / "metrics.json").read_text())
    assert metrics["n_val"] == 6
    assert 0.0 <= metrics["accuracy"] <= 100.0
    rows = (reports_dir / "confusion.csv").read_text().splitlines()
    assert rows[0] == "actual,predicted"
    assert len(rows) == 1 + 6
