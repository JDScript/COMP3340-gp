from dataclasses import dataclass, field
from xmlrpc.client import boolean
from omegaconf import OmegaConf, DictConfig, MISSING
from typing import cast, Optional


@dataclass
class _ClassObjectConfig:
    target: str = ""
    params: Optional[dict] = field(default_factory=dict)


@dataclass
class _ModelConfig:
    backbone: _ClassObjectConfig
    neck: _ClassObjectConfig
    head: _ClassObjectConfig


@dataclass
class _DatasetConfig:
    target: str
    params: dict = field(default_factory=dict)
    shuffle: Optional[bool] = True
    batch_size: Optional[int] = 8
    train_transforms: list[_ClassObjectConfig] = field(default_factory=list)
    val_transforms: list[_ClassObjectConfig] = field(default_factory=list)
    test_transforms: list[_ClassObjectConfig] = field(default_factory=list)


@dataclass
class _TrainerConfig:
    epochs: int
    optimizer: _ClassObjectConfig
    scheduler: _ClassObjectConfig


@dataclass
class Config(DictConfig):
    model: _ModelConfig
    dataset: _DatasetConfig
    trainer: _TrainerConfig


def load_config(
    conf_path="./configs/_base.yaml",
):
    schema = OmegaConf.structured(Config)
    base = OmegaConf.load("./configs/_base.yaml")
    conf = OmegaConf.load(conf_path)
    mergedConfig = cast(Config, OmegaConf.merge(schema, base, conf))
    return mergedConfig
