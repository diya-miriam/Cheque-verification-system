from pathlib import Path
import cv2

def load_cached_roi(cache_path: Path):
    if cache_path.exists():
        return cv2.imread(str(cache_path), cv2.IMREAD_GRAYSCALE)
    return None

def save_cached_roi(cache_path: Path, roi):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(cache_path), roi)