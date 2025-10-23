import os
from dataclasses import asdict, dataclass

import yaml

import constants


@dataclass(frozen=True)
class Config:
    lr: float = 0.001
    batch_size: int = 32
    epochs: int = 10
    data_url: str = "https://stnordaxoninternal.blob.core.windows.net/fish-dataset"
    image_count: int = 1000


def load_config(path: str | None = None) -> Config:
    cfg = asdict(Config())
    path = constants.CONFIG_FILE_PATH
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            file_cfg = yaml.safe_load(f) or {}
        if not isinstance(file_cfg, dict):
            raise TypeError("YAML must be a mapping of keys to values.")
        cfg.update(file_cfg)

    return Config(**cfg)


CONFIG = load_config()
