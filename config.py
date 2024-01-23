from dataclasses import dataclass, field
from omegaconf import OmegaConf, DictConfig, MISSING
from typing import cast, Optional


@dataclass
class _ClassObjectConfig:
    target: str
    params: dict = field(default_factory=dict)


@dataclass
class _DatasetConfig:
    path: str = MISSING
    split_id: str = MISSING
    batch_size: int = MISSING
    shuffle: bool = MISSING
    augmentations: list[_ClassObjectConfig] = field(default_factory=list)
    transforms: list[_ClassObjectConfig] = field(default_factory=list)


@dataclass
class _ModelConfig:
    backbone: _ClassObjectConfig
    classifier: _ClassObjectConfig


@dataclass
class _TrainerConfig:
    epochs: int
    optimizer: _ClassObjectConfig
    scheduler: _ClassObjectConfig


@dataclass(frozen=True)
class Config(DictConfig):
    device: str
    model: _ModelConfig
    trainer: _TrainerConfig
    dataset: _DatasetConfig
    ckpt: Optional[str] = None


def load_config(
    conf_path="./config.yaml",
):
    schema = OmegaConf.structured(Config)
    conf = OmegaConf.load(conf_path)
    mergedConfig = cast(Config, OmegaConf.merge(schema, conf, OmegaConf.from_cli()))
    return mergedConfig


if __name__ == "__main__":
    conf = load_config()
    print(conf)
