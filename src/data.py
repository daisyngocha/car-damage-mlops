from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def build_transforms(cfg: DictConfig, train: bool) -> transforms.Compose:
    normalize = transforms.Normalize(mean=list(cfg.mean), std=list(cfg.std))
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(cfg.img_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(cfg.resize),
            transforms.CenterCrop(cfg.img_size),
            transforms.ToTensor(),
            normalize,
        ]
    )


def build_dataset(cfg: DictConfig, split: str) -> datasets.ImageFolder:
    root = cfg.train_dir if split == "train" else cfg.val_dir
    return datasets.ImageFolder(root, build_transforms(cfg, train=split == "train"))


def build_loader(dataset, batch_size: int, num_workers: int, shuffle: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
