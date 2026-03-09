from __future__ import annotations

import numpy as np
import cv2
from loguru import logger
from src.utils.config_loader import preprocessing_cfg as cfg
import matplotlib.pyplot as plt
_br = cfg.background_removal

METHOD: str = _br.method
INVERT_IF_DARK: bool = _br.invert_if_dark_bg

# Otsu params
OTSU_BLUR: int = _br.otsu.blur_kernel_size
OTSU_MORPH: int = _br.otsu.morph_kernel_size
OTSU_ITER: int = _br.otsu.morph_iterations

# Adaptive params
ADAPTIVE_BLOCK: int = _br.adaptive.block_size
ADAPTIVE_C: int = _br.adaptive.c_constant


def remove_background(image: np.ndarray) -> np.ndarray:
    if not _br.enabled:
        logger.debug("Background removal disabled in config, skipping.")
        return image.astype(np.float32) / 255.0

    if INVERT_IF_DARK and _is_dark_background(image):
        image = cv2.bitwise_not(image)
        logger.debug("Inverted image (detected dark background).")

    if METHOD == "otsu":
        gray = _otsu_grayscale(image)
    elif METHOD == "adaptive":
        gray = _adaptive_grayscale(image)
    elif METHOD == "grabcut":
        gray = _grabcut_grayscale(image)
    else:
        raise ValueError(f"Unknown background removal method: {METHOD}")

    return gray 


def _otsu_threshold(image: np.ndarray) -> np.ndarray: 
    blurred = cv2.GaussianBlur(image, (OTSU_BLUR, OTSU_BLUR), 0) 
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (OTSU_MORPH, OTSU_MORPH)) 
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=OTSU_ITER) 
    return binary

def _otsu_grayscale(image: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (OTSU_BLUR, OTSU_BLUR), 0)

    _, mask = cv2.threshold(
        blurred, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (OTSU_MORPH, OTSU_MORPH)
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, kernel, iterations=OTSU_ITER
    )

    return _apply_mask_white_bg(image, mask)


def _adaptive_threshold(image: np.ndarray) -> np.ndarray:
    binary = cv2.adaptiveThreshold(
        image,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=ADAPTIVE_BLOCK,
        C=ADAPTIVE_C,
    )
    return binary

def _adaptive_grayscale(image: np.ndarray) -> np.ndarray:
    mask = cv2.adaptiveThreshold(
        image, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        ADAPTIVE_BLOCK,
        ADAPTIVE_C,
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return _apply_mask_white_bg(image, mask)

def _grabcut_grayscale(image: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    h, w = image.shape

    rect = (w // 10, h // 10, w * 8 // 10, h * 8 // 10)

    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(
        rgb, mask, rect,
        bgd_model, fgd_model,
        5, cv2.GC_INIT_WITH_RECT
    )

    fg_mask = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255, 0
    ).astype(np.uint8)

    return _apply_mask_white_bg(image, fg_mask)


def _grabcut(image: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    h, w = image.shape
    margin_x, margin_y = w // 10, h // 10
    rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)
    mask = np.zeros((h, w), dtype=np.uint8)
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)
    cv2.grabCut(rgb, mask, rect, bgd_model, fgd_model, iterCount=5, mode=cv2.GC_INIT_WITH_RECT)
    grab_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    return grab_mask

def _is_dark_background(image: np.ndarray) -> bool:
    return float(np.mean(image)) < 127

def _apply_mask_white_bg(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    output = image.copy()
    output[mask == 0] = 255 
    return output


#2
# from __future__ import annotations

# import numpy as np
# import cv2
# from loguru import logger
# from src.utils.config_loader import preprocessing_cfg as cfg

# _br = cfg.background_removal

# METHOD: str = _br.method
# INVERT_IF_DARK: bool = _br.invert_if_dark_bg

# # Otsu params
# OTSU_BLUR: int = _br.otsu.blur_kernel_size
# OTSU_MORPH: int = _br.otsu.morph_kernel_size
# OTSU_ITER: int = _br.otsu.morph_iterations

# # Adaptive params
# ADAPTIVE_BLOCK: int = _br.adaptive.block_size
# ADAPTIVE_C: int = _br.adaptive.c_constant


# def remove_background(image: np.ndarray) -> np.ndarray:
#     if not _br.enabled:
#         logger.debug("Background removal disabled in config, skipping.")
#         return image.astype(np.float32) / 255.0

#     if INVERT_IF_DARK and _is_dark_background(image):
#         image = cv2.bitwise_not(image)
#         logger.debug("Inverted image (detected dark background).")

#     if METHOD == "otsu":
#         gray = _otsu_grayscale(image)
#     elif METHOD == "adaptive":
#         gray = _adaptive_grayscale(image)
#     elif METHOD == "grabcut":
#         gray = _grabcut_grayscale(image)
#     else:
#         raise ValueError(f"Unknown background removal method: {METHOD}")

#     return gray


# def _otsu_threshold(image: np.ndarray) -> np.ndarray:
#     blurred = cv2.GaussianBlur(image, (OTSU_BLUR, OTSU_BLUR), 0)
#     _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (OTSU_MORPH, OTSU_MORPH))
#     binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=OTSU_ITER)
#     binary = _keep_signature_components(binary)
#     return binary


# def _otsu_grayscale(image: np.ndarray) -> np.ndarray:
#     blurred = cv2.GaussianBlur(image, (OTSU_BLUR, OTSU_BLUR), 0)

#     _, mask = cv2.threshold(
#         blurred,
#         0,
#         255,
#         cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
#     )

#     kernel = cv2.getStructuringElement(
#         cv2.MORPH_RECT, (OTSU_MORPH, OTSU_MORPH)
#     )
#     mask = cv2.morphologyEx(
#         mask,
#         cv2.MORPH_CLOSE,
#         kernel,
#         iterations=OTSU_ITER
#     )

#     mask = _keep_signature_components(mask)

#     return _apply_mask_white_bg(image, mask)


# def _adaptive_threshold(image: np.ndarray) -> np.ndarray:
#     binary = cv2.adaptiveThreshold(
#         image,
#         maxValue=255,
#         adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         thresholdType=cv2.THRESH_BINARY_INV,
#         blockSize=ADAPTIVE_BLOCK,
#         C=ADAPTIVE_C,
#     )
#     binary = _keep_signature_components(binary)
#     return binary


# def _adaptive_grayscale(image: np.ndarray) -> np.ndarray:
#     mask = cv2.adaptiveThreshold(
#         image,
#         255,
#         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#         cv2.THRESH_BINARY_INV,
#         ADAPTIVE_BLOCK,
#         ADAPTIVE_C,
#     )

#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

#     mask = _keep_signature_components(mask)

#     return _apply_mask_white_bg(image, mask)


# def _grabcut_grayscale(image: np.ndarray) -> np.ndarray:
#     rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
#     h, w = image.shape

#     rect = (w // 10, h // 10, w * 8 // 10, h * 8 // 10)

#     mask = np.zeros((h, w), np.uint8)
#     bgd_model = np.zeros((1, 65), np.float64)
#     fgd_model = np.zeros((1, 65), np.float64)

#     cv2.grabCut(
#         rgb, mask, rect,
#         bgd_model, fgd_model,
#         5, cv2.GC_INIT_WITH_RECT
#     )

#     fg_mask = np.where(
#         (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
#         255, 0
#     ).astype(np.uint8)

#     fg_mask = _keep_signature_components(fg_mask)

#     return _apply_mask_white_bg(image, fg_mask)


# def _grabcut(image: np.ndarray) -> np.ndarray:
#     rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
#     h, w = image.shape
#     margin_x, margin_y = w // 10, h // 10
#     rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)
#     mask = np.zeros((h, w), dtype=np.uint8)
#     bgd_model = np.zeros((1, 65), dtype=np.float64)
#     fgd_model = np.zeros((1, 65), dtype=np.float64)
#     cv2.grabCut(rgb, mask, rect, bgd_model, fgd_model, iterCount=5, mode=cv2.GC_INIT_WITH_RECT)
#     grab_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
#     grab_mask = _keep_signature_components(grab_mask)
#     return grab_mask


# def _keep_signature_components(mask: np.ndarray) -> np.ndarray:
#     """
#     Keep only meaningful connected components likely to belong to the signature.
#     This helps remove small printed text, random specks, and minor artifacts.
#     """
#     num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

#     # Only background present
#     if num_labels <= 1:
#         return mask

#     components = []
#     for label in range(1, num_labels):
#         area = int(stats[label, cv2.CC_STAT_AREA])
#         x = int(stats[label, cv2.CC_STAT_LEFT])
#         y = int(stats[label, cv2.CC_STAT_TOP])
#         w = int(stats[label, cv2.CC_STAT_WIDTH])
#         h = int(stats[label, cv2.CC_STAT_HEIGHT])

#         components.append({
#             "label": label,
#             "area": area,
#             "x": x,
#             "y": y,
#             "w": w,
#             "h": h,
#         })

#     if not components:
#         return mask

#     max_area = max(c["area"] for c in components)

#     # Keep components that are reasonably significant relative to the biggest one
#     # and also not trivially tiny.
#     kept = [
#         c for c in components
#         if c["area"] >= max(30, int(0.08 * max_area))
#     ]

#     # Fallback: if filtering removes everything useful, keep the largest component
#     if not kept:
#         largest = max(components, key=lambda c: c["area"])
#         kept = [largest]

#     cleaned = np.zeros_like(mask)
#     for comp in kept:
#         cleaned[labels == comp["label"]] = 255

#     return cleaned


# def _is_dark_background(image: np.ndarray) -> bool:
#     return float(np.mean(image)) < 127


# def _apply_mask_white_bg(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
#     output = image.copy()
#     output[mask == 0] = 255
#     return output