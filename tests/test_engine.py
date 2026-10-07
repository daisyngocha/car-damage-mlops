import pytest
import torch
from conftest import tiny_model
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.engine import predict, train_one_epoch

CPU = torch.device("cpu")


def test_train_one_epoch_updates_weights(tiny_loader):
    torch.manual_seed(0)
    model = tiny_model()
    before = [p.detach().clone() for p in model.parameters()]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    res = train_one_epoch(model, tiny_loader, nn.CrossEntropyLoss(), optimizer, CPU)

    assert model.training
    assert any(not torch.equal(b, p) for b, p in zip(before, model.parameters()))
    assert res["loss"] > 0
    assert 0.0 <= res["acc"] <= 100.0


def test_predict_returns_labels_in_eval_mode(tiny_loader):
    model = tiny_model()

    res = predict(model, tiny_loader, nn.CrossEntropyLoss(), CPU)

    assert not model.training
    assert len(res["y_true"]) == len(res["y_pred"]) == 8
    assert 0.0 <= res["acc"] <= 100.0


def test_predict_perfect_model_scores_100():
    # A model that echoes one-hot targets must hit 100% accuracy.
    targets = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
    inputs = nn.functional.one_hot(targets, 3).float()
    loader = DataLoader(TensorDataset(inputs, targets), batch_size=4)

    res = predict(nn.Identity(), loader, nn.CrossEntropyLoss(), CPU)

    assert res["acc"] == pytest.approx(100.0)
    assert res["y_pred"] == targets.tolist()
