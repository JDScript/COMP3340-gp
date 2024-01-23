from config import Config
from utils import retrieve_class_from_string
from torchvision.transforms import Compose, RandomResizedCrop
from torch.utils.data.dataloader import DataLoader
from .flowers17 import Flowers17


def instentiate_dataloader(cfg: Config):
    augmentations = Compose(
        [
            retrieve_class_from_string(aug.target)(**aug.params)
            for aug in cfg.dataset.augmentations
        ]
    )
    transforms = Compose(
        [
            retrieve_class_from_string(transform.target)(**transform.params)
            for transform in cfg.dataset.transforms
        ]
    )

    train_dataset = Flowers17(
        cfg.dataset.path, split="train", download=True, transform=Compose([augmentations, transforms])
    )
    val_dataset = Flowers17(cfg.dataset.path, split="val", transform=transforms)
    test_dataset = Flowers17(cfg.dataset.path, split="test", transform=transforms)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        shuffle=cfg.dataset.shuffle,
        batch_size=cfg.dataset.batch_size,
        
    )
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=cfg.dataset.batch_size)
    test_dataloader = DataLoader(
        dataset=test_dataset, batch_size=cfg.dataset.batch_size
    )

    return train_dataloader, val_dataloader, test_dataloader
