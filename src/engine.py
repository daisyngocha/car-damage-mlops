import torch
from torch import nn


def train_one_epoch(model: nn.Module, loader, criterion, optimizer, device) -> dict:
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for images, target in loader:
        images, target = images.to(device), target.to(device)
        output = model(images)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * target.size(0)
        correct += (output.argmax(1) == target).sum().item()
        total += target.size(0)
    return {"loss": loss_sum / total, "acc": 100.0 * correct / total}


@torch.no_grad()
def predict(model: nn.Module, loader, criterion, device) -> dict:
    model.eval()
    loss_sum, total = 0.0, 0
    y_true, y_pred = [], []
    for images, target in loader:
        images, target = images.to(device), target.to(device)
        output = model(images)
        loss_sum += criterion(output, target).item() * target.size(0)
        total += target.size(0)
        y_true += target.tolist()
        y_pred += output.argmax(1).tolist()
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    return {
        "loss": loss_sum / total,
        "acc": 100.0 * correct / total,
        "y_true": y_true,
        "y_pred": y_pred,
    }
