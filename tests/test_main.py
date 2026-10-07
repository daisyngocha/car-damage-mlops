"""Unit tests for the helpers and train/validate loop in main.py (CPU only)."""

import argparse
import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import main


@pytest.fixture
def cpu_args():
    return argparse.Namespace(
        no_accel=True,
        print_freq=1,
        gpu=None,
        distributed=False,
        world_size=1,
        batch_size=4,
        workers=0,
    )


@pytest.fixture
def tiny_loader():
    torch.manual_seed(0)
    images = torch.randn(8, 3, 8, 8)
    targets = torch.randint(0, 6, (8,))
    return DataLoader(TensorDataset(images, targets), batch_size=4)


def tiny_model(num_classes=6):
    return nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, num_classes))


# --- accuracy -----------------------------------------------------------------


def test_accuracy_all_correct():
    output = torch.tensor([[0.1, 0.9, 0.0], [0.8, 0.1, 0.1]])
    target = torch.tensor([1, 0])
    (acc1,) = main.accuracy(output, target, topk=(1,))
    assert acc1.item() == pytest.approx(100.0)


def test_accuracy_top1_vs_topk():
    # Sample 0: correct at top-1. Sample 1: true class ranked 2nd.
    output = torch.tensor([[0.7, 0.2, 0.1], [0.6, 0.3, 0.1]])
    target = torch.tensor([0, 1])
    acc1, acc2 = main.accuracy(output, target, topk=(1, 2))
    assert acc1.item() == pytest.approx(50.0)
    assert acc2.item() == pytest.approx(100.0)


def test_accuracy_none_correct():
    output = torch.tensor([[0.9, 0.1, 0.0], [0.9, 0.1, 0.0]])
    target = torch.tensor([2, 2])
    (acc1,) = main.accuracy(output, target, topk=(1,))
    assert acc1.item() == pytest.approx(0.0)


# --- AverageMeter -------------------------------------------------------------


def test_average_meter_weighted_average():
    meter = main.AverageMeter("Loss", use_accel=False)
    meter.update(1.0, n=2)
    meter.update(4.0, n=1)
    assert meter.val == 4.0
    assert meter.sum == pytest.approx(6.0)
    assert meter.count == 3
    assert meter.avg == pytest.approx(2.0)


def test_average_meter_reset():
    meter = main.AverageMeter("Loss", use_accel=False)
    meter.update(5.0, n=3)
    meter.reset()
    assert (meter.val, meter.avg, meter.sum, meter.count) == (0, 0, 0, 0)


@pytest.mark.parametrize(
    ("summary_type", "expected"),
    [
        (main.Summary.NONE, ""),
        (main.Summary.AVERAGE, "Acc 2.000"),
        (main.Summary.SUM, "Acc 6.000"),
        (main.Summary.COUNT, "Acc 3.000"),
    ],
)
def test_average_meter_summary(summary_type, expected):
    meter = main.AverageMeter("Acc", False, ":6.2f", summary_type)
    meter.update(1.0, n=2)
    meter.update(4.0, n=1)
    assert meter.summary() == expected


def test_average_meter_str():
    meter = main.AverageMeter("Loss", False, ":.2f")
    meter.update(1.5)
    assert str(meter) == "Loss 1.50 (1.50)"


def test_average_meter_invalid_summary_type():
    meter = main.AverageMeter("Acc", False, summary_type="bogus")
    with pytest.raises(ValueError):
        meter.summary()


# --- ProgressMeter ------------------------------------------------------------


def test_progress_meter_display(capsys):
    meter = main.AverageMeter("Loss", False, ":.1f")
    meter.update(2.0)
    progress = main.ProgressMeter(100, [meter], prefix="Epoch: [0]")
    progress.display(7)
    assert capsys.readouterr().out.strip() == "Epoch: [0][  7/100]\tLoss 2.0 (2.0)"


def test_progress_meter_display_summary(capsys):
    meter = main.AverageMeter("Acc@1", False, summary_type=main.Summary.AVERAGE)
    meter.update(50.0)
    main.ProgressMeter(1, [meter]).display_summary()
    assert capsys.readouterr().out.strip() == "* Acc@1 50.000"


# --- save_checkpoint ----------------------------------------------------------


def test_save_checkpoint_not_best(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    main.save_checkpoint({"epoch": 1}, is_best=False)
    assert (tmp_path / "checkpoint.pth.tar").exists()
    assert not (tmp_path / "model_best.pth.tar").exists()


def test_save_checkpoint_best_copies_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    main.save_checkpoint({"epoch": 3, "best_acc1": 42.0}, is_best=True)
    best = torch.load(tmp_path / "model_best.pth.tar")
    assert best == {"epoch": 3, "best_acc1": 42.0}


# --- train / validate smoke tests --------------------------------------------


def test_train_one_epoch_updates_weights(cpu_args, tiny_loader):
    torch.manual_seed(0)
    model = tiny_model()
    before = [p.detach().clone() for p in model.parameters()]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    main.train(
        tiny_loader,
        model,
        nn.CrossEntropyLoss(),
        optimizer,
        epoch=0,
        device=torch.device("cpu"),
        args=cpu_args,
    )

    assert model.training
    assert any(not torch.equal(b, p) for b, p in zip(before, model.parameters()))


def test_validate_returns_top1_in_range(cpu_args, tiny_loader):
    torch.manual_seed(0)
    model = tiny_model()

    acc1 = main.validate(tiny_loader, model, nn.CrossEntropyLoss(), cpu_args)

    assert not model.training
    assert 0.0 <= float(acc1) <= 100.0


def test_validate_perfect_model_scores_100(cpu_args):
    # A model that echoes one-hot targets must hit 100% top-1.
    targets = torch.tensor([0, 1, 2, 3, 4, 5, 0, 1])
    inputs = nn.functional.one_hot(targets, 6).float()
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=4)

    acc1 = main.validate(loader, nn.Identity(), nn.CrossEntropyLoss(), cpu_args)

    assert float(acc1) == pytest.approx(100.0)
