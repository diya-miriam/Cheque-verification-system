from ultralytics import YOLO
import torch
from pathlib import Path
#MODEL_PATH = "src/models/checkpoints/roi_model.pt"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "src" / "models" / "checkpoints" / "roi_model.pt"

_roi_model = None

def get_roi_model():
    global _roi_model
    if _roi_model is None:
        _roi_model = YOLO(MODEL_PATH)
    return _roi_model