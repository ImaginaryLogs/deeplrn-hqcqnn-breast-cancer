import yaml
from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True) 
class SplitConfig:
    train_split: float
    validation_split: float

@dataclass(frozen=True)
class TrainingConfig:
    split: SplitConfig


@dataclass(frozen=True)
class ClahePreprocessConfig:
    clip_limit: float
    tile_grid: int

# --------------------------------------------------

@dataclass(frozen=True)
class GlobalConfig:
    seed: int
    target_size: int
    training: TrainingConfig
    clahe: ClahePreprocessConfig




with open("configs/base.yml", "r") as file:
    raw_config = yaml.safe_load(file)




CONFIG: Final[GlobalConfig] = GlobalConfig(
    seed=raw_config["seed"],
    target_size=raw_config["target_size"],
    training=TrainingConfig(
        split=SplitConfig(**raw_config["training"]["split"])
    ),
    clahe=ClahePreprocessConfig(**raw_config["clahe"])
)
