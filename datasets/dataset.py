from config import Config
from utils import retrieve_class_from_string
from torchvision.transforms import Compose, RandomResizedCrop
from torch.utils.data.dataloader import DataLoader
from os import cpu_count


def instentiate_dataloader(cfg: Config):
    num_workers = cpu_count()
    num_workers = num_workers if num_workers is not None else 1
    train_transforms = Compose(
        [
            retrieve_class_from_string(aug.target)(**aug.params)
            for aug in cfg.dataset.train_transforms
        ]
    )
    val_transforms = Compose(
        [
            retrieve_class_from_string(transform.target)(**transform.params)
            for transform in cfg.dataset.val_transforms
        ]
    )
    test_transforms = Compose(
        [
            retrieve_class_from_string(transform.target)(**transform.params)
            for transform in cfg.dataset.test_transforms
        ]
    )

    dataset_cls = retrieve_class_from_string(cfg.dataset.target)

    train_dataset = dataset_cls(
        **cfg.dataset.params, split="train", download=True, transform=train_transforms
    )
    val_dataset = dataset_cls(
        **cfg.dataset.params, split="val", transform=val_transforms
    )
    test_dataset = dataset_cls(
        **cfg.dataset.params, split="test", transform=test_transforms
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        shuffle=cfg.dataset.shuffle,
        batch_size=cfg.dataset.batch_size,
        num_workers=num_workers,
        persistent_workers=True,
    )
    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.dataset.batch_size,
        num_workers=num_workers,
        persistent_workers=True,
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=cfg.dataset.batch_size,
        num_workers=num_workers,
        persistent_workers=True,
    )

    return train_dataloader, val_dataloader, test_dataloader
