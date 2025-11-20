# Standard Imports
from pathlib import Path
from typing import Any

# Third Party Imports
from omegaconf import OmegaConf

# Internal Imports
from .config import Config


def load_config(config_path: str | Path) -> tuple[Config | dict[str, Any]]:
    # Load the config
    cfg = OmegaConf.load(config_path)
    cfg_raw: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)
    config = Config(**cfg_raw)
    return config, cfg_raw
