from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import cv2
from PIL import Image
from loguru import logger

from src.utils.config_loader import preprocessing_cfg as cfg
from src.preprocessing.roi_model import get_roi_model

_roi = cfg.roi_extraction

METHOD: str = _roi.method

# Fixed ratio params
FR_X0: float = _roi.fixed_ratio.x_start
FR_Y0: float = _roi.fixed_ratio.y_start
FR_X1: float = _roi.fixed_ratio.x_end
FR_Y1: float = _roi.fixed_ratio.y_end

# Contour params
CONTOUR_MIN_AREA: int = _roi.contour_detection.min_area
CONTOUR_PAD: int = _roi.contour_detection.padding_px

# Output params
OUT_W: int = _roi.output.width
OUT_H: int = _roi.output.height
KEEP_ASPECT: bool = _roi.output.keep_aspect_ratio
PAD_COLOR: int = _roi.output.pad_color


def extract_roi(image: np.ndarray) -> np.ndarray:
    if not _roi.enabled:
        logger.debug("ROI extraction disabled in config, skipping.")
        return image

    if METHOD == "fixed_ratio":
        crop = _fixed_ratio_crop(image)
    elif METHOD == "contour_detection":
        crop = _contour_crop(image)
    elif(METHOD == "model"):
        crop = _model_based_roi(image)
    else:
        raise ValueError(f"Unknown ROI method: {METHOD}")

    resized = _resize_roi(crop)
    logger.debug(
        f"ROI extracted ({METHOD}): crop={crop.shape[:2]} → output={resized.shape[:2]}"
    )
    return resized


def save_roi(
    roi: np.ndarray,
    cheque_id: str,
    output_dir: Optional[str] = None,
) -> Path:
    out_dir = Path(output_dir or cfg.output.final_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fmt = cfg.output.format
    fname = out_dir / f"{cheque_id}_signature.{fmt}"

    pil = Image.fromarray(roi, mode="L")
    save_kwargs = {}
    if fmt == "jpg":
        save_kwargs["quality"] = cfg.output.jpeg_quality
    pil.save(str(fname), **save_kwargs)
    logger.info(f"ROI saved: {fname}")
    return fname


def _fixed_ratio_crop(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    x0 = int(FR_X0 * w)
    y0 = int(FR_Y0 * h)
    x1 = int(FR_X1 * w)
    y1 = int(FR_Y1 * h)
    return image[y0:y1, x0:x1]


def _contour_crop(image: np.ndarray) -> np.ndarray:
    search = _fixed_ratio_crop(image)
    _, binary = cv2.threshold(search, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        logger.warning("Contour ROI: no contours found. Falling back to fixed ratio.")
        return search

    valid = [c for c in contours if cv2.contourArea(c) >= CONTOUR_MIN_AREA]
    if not valid:
        logger.warning("Contour ROI: no contour meets min_area. Falling back to fixed ratio.")
        return search
    all_pts = np.vstack(valid)
    x, y, w, h = cv2.boundingRect(all_pts)
    sh, sw = search.shape[:2]
    x = max(0, x - CONTOUR_PAD)
    y = max(0, y - CONTOUR_PAD)
    w = min(sw - x, w + 2 * CONTOUR_PAD)
    h = min(sh - y, h + 2 * CONTOUR_PAD)

    return search[y : y + h, x : x + w]


def _resize_roi(crop: np.ndarray) -> np.ndarray:
    if KEEP_ASPECT:
        return _resize_with_pad(crop, OUT_W, OUT_H)
    pil = Image.fromarray(crop, mode="L")
    return np.array(pil.resize((OUT_W, OUT_H), Image.LANCZOS), dtype=np.uint8)


def _resize_with_pad(
    image: np.ndarray,
    target_w: int,
    target_h: int,
) -> np.ndarray:
    h, w = image.shape
    scale = min(target_w / w, target_h / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    pil = Image.fromarray(image, mode="L")
    resized = np.array(pil.resize((new_w, new_h), Image.LANCZOS), dtype=np.uint8)

    canvas = np.full((target_h, target_w), PAD_COLOR, dtype=np.uint8)
    y_off = (target_h - new_h) // 2
    x_off = (target_w - new_w) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
    return canvas

def _model_based_roi(image):
    model = get_roi_model()
    results = model(ensure_rgb_for_model(image), conf=0.4, iou=0.5)

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return image  

    x1, y1, x2, y2 = boxes.xyxy[0].int().tolist()
    return image[y1:y2, x1:x2]


def ensure_rgb_for_model(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return np.repeat(img[:, :, None], 3, axis=2)
    return img
