import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import torchvision.transforms as T

from src.preprocessing.pipeline import PreprocessingPipeline
from src.dataset.cache_utils import load_cached_roi, save_cached_roi


class SiamesePairDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        self.pipeline = PreprocessingPipeline()
        self.transform = T.Compose([
            T.Resize((224, 224))
        ])

        self.raw_root = Path("src/data/raw")
        self.cache_root = Path("src/data/signatures")

    def _get_image(self, rel_path: str):
        raw_path = self.raw_root / rel_path
        cache_path = self.cache_root / rel_path

        roi = load_cached_roi(cache_path)
        if roi is None:
            result = self.pipeline.run(raw_path)
            if not result.success:
                raise RuntimeError(f"Preprocessing failed for {rel_path}")
            roi = result.roi
            save_cached_roi(cache_path, roi)

        roi = roi.astype(np.float32)
        roi = roi / 127.5 - 1.0
        roi = torch.from_numpy(roi).unsqueeze(0)
        return roi

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img1 = self.transform(self._get_image(row.image1))
        img2 = self.transform(self._get_image(row.image2))
        label = torch.tensor(float(row.label), dtype=torch.float32)
        return img1, img2, label

    def __len__(self):
        return len(self.df)