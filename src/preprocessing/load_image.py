from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from loguru import logger

from src.utils.config_loader import preprocessing_cfg as cfg

COLOR_MODE: str = cfg.image.color_mode          # "grayscale" | "rgb"
DEFAULT_DPI: int = cfg.dpi_normalization.default_assumed_dpi
TARGET_DPI: int = cfg.image.target_dpi


def load_image(
    path: str | Path,
    color_mode: Optional[str] = None,
) -> Tuple[np.ndarray, dict]:

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    mode = color_mode or COLOR_MODE
    pil_img = Image.open(path)
    dpi = _extract_dpi(pil_img)

    if mode == "grayscale":
        pil_img = pil_img.convert("L")
    else:
        pil_img = pil_img.convert("RGB")

    image = np.array(pil_img, dtype=np.uint8)

    metadata = {
        "path": str(path),
        "dpi": dpi,
        "original_size": pil_img.size,   
        "color_mode": mode,
    }

    logger.debug(
        f"Loaded {path.name} | size={pil_img.size} | dpi={dpi} | mode={mode}"
    )
    return image, metadata

def load_image_cv2(path: str | Path) -> np.ndarray:
    image, _ = load_image(path)
    return image


def _extract_dpi(pil_img: Image.Image) -> int:
    try:
        dpi_info = pil_img.info.get("dpi") or pil_img.info.get("jfif_density")
        if dpi_info:
            dpi = int(dpi_info[0]) if isinstance(dpi_info, (tuple, list)) else int(dpi_info)
            if dpi > 0:
                return dpi
    except Exception:
        pass
    return DEFAULT_DPI


def validate_image(image: np.ndarray) -> bool:
    """Basic sanity check: non-empty array with expected bit depth."""
    if image is None or image.size == 0:
        return False
    if image.dtype != np.uint8:
        return False
    return True
