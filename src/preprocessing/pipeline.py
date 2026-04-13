"""
src/preprocessing/pipeline.py
Orchestrates the full preprocessing pipeline:
    load → dpi_normalize → resolution → skew → perspective → noise → background → roi
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from src.utils.config_loader import preprocessing_cfg, refsig_preprocessing
from src.preprocessing.load_image import load_image
from src.preprocessing.dpi_normalization import normalize_dpi
from src.preprocessing.resolution_enforcement import enforce_resolution
from src.preprocessing.skew_correction import correct_skew
from src.preprocessing.perspective_correction import correct_perspective
from src.preprocessing.background_removal import remove_background
from src.preprocessing.noise_removal import remove_noise
from src.preprocessing.roi_extraction import extract_roi, save_roi


@dataclass
class PreprocessingResult:
    cheque_id: str
    roi: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)
    stage_times_ms: dict = field(default_factory=dict)
    intermediates: dict = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None


class PreprocessingPipeline:

    def __init__(self, use_reference: bool = False):
        # Store config on self — do NOT use global
        if use_reference:
            self.cfg = refsig_preprocessing
            logger.info("PreprocessingPipeline initialised with reference_signature_preprocessing.yaml")
        else:
            self.cfg = preprocessing_cfg
            logger.info("PreprocessingPipeline initialised with preprocessing.yaml")

        self.SAVE_INTERMEDIATES = self.cfg.output.save_intermediates
        self.INTERMEDIATE_DIR = self.cfg.output.intermediate_dir

    def run(self, image_path: str | Path, cheque_id: Optional[str] = None) -> PreprocessingResult:
        image_path = Path(image_path)
        cheque_id = cheque_id or image_path.stem
        result = PreprocessingResult(cheque_id=cheque_id)
        timings = {}
        intermediates = {}

        try:
            # Stage 1: Load
            t = time.perf_counter()
            image, metadata = load_image(image_path)
            timings["load"] = _ms(t)
            result.metadata = metadata
            intermediates["load"] = image.copy()

            # Stage 2: DPI Normalisation
            t = time.perf_counter()
            image = normalize_dpi(image, source_dpi=metadata["dpi"])
            timings["dpi_normalize"] = _ms(t)
            intermediates["dpi_normalize"] = image.copy()

            # Stage 3: Resolution Enforcement
            t = time.perf_counter()
            image = enforce_resolution(image)
            timings["resolution"] = _ms(t)
            intermediates["resolution"] = image.copy()

            # Stage 4: Skew Correction
            t = time.perf_counter()
            image = correct_skew(image)
            timings["skew"] = _ms(t)
            intermediates["skew"] = image.copy()

            # Stage 5: Perspective Correction
            t = time.perf_counter()
            image = correct_perspective(image)
            timings["perspective"] = _ms(t)
            intermediates["perspective"] = image.copy()

            # Stage 6: Noise Removal
            t = time.perf_counter()
            image = remove_noise(image)
            timings["noise"] = _ms(t)
            intermediates["noise"] = image.copy()

            # Stage 7: Background Removal
            t = time.perf_counter()
            image = remove_background(image)
            timings["background"] = _ms(t)
            intermediates["background"] = image.copy()

            # Stage 8: ROI Extraction — LAST, on clean image
            t = time.perf_counter()
            image = extract_roi(image)
            timings["roi"] = _ms(t)
            intermediates["roi"] = image.copy()

            # Ensure uint8 output
            if image.dtype != np.uint8:
                image = (image * 255).clip(0, 255).astype(np.uint8)

            result.roi = image
            result.stage_times_ms = timings

            if self.SAVE_INTERMEDIATES:
                result.intermediates = intermediates
                self._save_intermediates(cheque_id, intermediates)

        except Exception as exc:
            logger.error(f"[{cheque_id}] Pipeline failed: {exc}")
            result.success = False
            result.error = str(exc)
            raise

        return result

    def run_batch(
        self,
        image_paths: list[str | Path],
        save_rois: bool = True,
    ) -> list[PreprocessingResult]:
        results = []
        for path in image_paths:
            res = self.run(path)
            if res.success and save_rois:
                save_roi(res.roi, res.cheque_id)
            results.append(res)

        succeeded = sum(1 for r in results if r.success)
        logger.info(f"Batch complete: {succeeded}/{len(results)} succeeded")
        return results

    def _save_intermediates(self, cheque_id: str, intermediates: dict) -> None:
        import cv2
        out_dir = Path(self.INTERMEDIATE_DIR) / cheque_id
        out_dir.mkdir(parents=True, exist_ok=True)
        for stage, img in intermediates.items():
            fpath = out_dir / f"{stage}.png"
            cv2.imwrite(str(fpath), img)
        logger.debug(f"Intermediates saved to {out_dir}")


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000