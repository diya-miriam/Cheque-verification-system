from __future__ import annotations

import sys
from pathlib import Path

import mlflow
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logging import setup_logging
from src.utils.config_loader import preprocessing_cfg, model_cfg
from src.dataset.siamese_dataset import SiamesePairDataset
from src.utils.csv_loader import load_pairs_csv
from src.utils.split import stratified_split

load_dotenv()
setup_logging()


def main() -> None:
    # ---------------- MLflow setup ----------------
    mlflow.set_tracking_uri(model_cfg.mlflow.tracking_uri)
    mlflow.set_experiment(model_cfg.mlflow.experiment_name)

    with mlflow.start_run(tags=dict(model_cfg.mlflow.run_tags)) as run:
        logger.info(f"MLflow run started: {run.info.run_id}")

        mlflow.log_params({
            "target_dpi": preprocessing_cfg.image.target_dpi,
            "color_mode": preprocessing_cfg.image.color_mode,
            "skew_method": preprocessing_cfg.skew_correction.method,
            "bg_removal_method": preprocessing_cfg.background_removal.method,
            "roi_method": preprocessing_cfg.roi_extraction.method,
            "roi_output_w": preprocessing_cfg.roi_extraction.output.width,
            "roi_output_h": preprocessing_cfg.roi_extraction.output.height,
        })

        # ---------------- Load CSV ----------------
        df = load_pairs_csv("/app/src/data/csv/data.csv")
        logger.info(f"Loaded CSV with {len(df)} pairs")

        train, val, test = stratified_split(df)

        logger.info(
            f"Split sizes → Train: {len(train)}, "
            f"Val: {len(val)}, Test: {len(test)}"
        )

        train_ds = SiamesePairDataset(train)
        val_ds   = SiamesePairDataset(val)
        test_ds  = SiamesePairDataset(test)

        train_loader = DataLoader(
            train_ds,
            batch_size=model_cfg.training.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=model_cfg.training.batch_size,
            shuffle=False,
            num_workers=4
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=model_cfg.training.batch_size,
            shuffle=False,
            num_workers=4
        )

        logger.success("Datasets and DataLoaders initialized")

        img1, img2, label = next(iter(train_loader))
        logger.info(
            f"Sample batch shapes → img1: {img1.shape}, img2: {img2.shape}"
        )


        logger.info("Model training stage: TODO")
        logger.info("Evaluation stage: TODO")

        logger.success(f"Pipeline run complete. MLflow run ID: {run.info.run_id}")

if __name__ == "__main__":
    main()