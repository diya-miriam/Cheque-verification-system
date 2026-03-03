from pathlib import Path
from typing import Any

import yaml
from omegaconf import DictConfig, OmegaConf

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"


def _load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def load_config(name: str) -> DictConfig:
    path = _CONFIG_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    raw = _load_yaml(path)
    return OmegaConf.create(raw)


def get(config: DictConfig, key: str, default: Any = None) -> Any:
    """Safe dot-path accessor with fallback default."""
    try:
        return OmegaConf.select(config, key, default=default)
    except Exception:
        return default


preprocessing_cfg: DictConfig = load_config("preprocessing")
augmentation_cfg: DictConfig = load_config("augmentation")
model_cfg: DictConfig = load_config("model")
training_cfg: DictConfig = load_config("training")
