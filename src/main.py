from __future__ import annotations

import sys
from pathlib import Path
import torch
import mlflow
import pandas as pd
from dotenv import load_dotenv
from loguru import logger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.logging import setup_logging
from src.utils.config_loader import preprocessing_cfg, model_cfg,training_cfg
from src.dataset.siamese_dataset import SiamesePairDataset
from src.utils.csv_loader import load_pairs_csv
from src.utils.split import writer_disjoint_split
from src.models.siamese_network import SiameseNetwork
from src.training.trainer import train_model

load_dotenv()
setup_logging()


def main() -> None:
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

        df = load_pairs_csv("/app/src/data/csv/data.csv")
        logger.info(f"Loaded CSV with {len(df)} pairs")

        train, val, test = writer_disjoint_split(df)

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
            num_workers=0,
            pin_memory=False
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=model_cfg.training.batch_size,
            shuffle=False,
            num_workers=0
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=model_cfg.training.batch_size,
            shuffle=False,
            num_workers=0
        )

        logger.success("Datasets and DataLoaders initialized")

        img1, img2, label = next(iter(train_loader))
        logger.info(
            f"Sample batch shapes → img1: {img1.shape}, img2: {img2.shape}"
        )

        logger.info("Model training stage")

        model = SiameseNetwork(
            embedding_size=training_cfg.model.embedding_size
        )
        checkpoint = torch.load(training_cfg.paths.pretrained_model, map_location="cpu")

        model.cnn.load_state_dict(checkpoint["cnn"], strict=True)
        model.fc.load_state_dict(checkpoint["fc"], strict=True)

        model.eval()

        margins = training_cfg.loss.margins
        for margin in margins:
            logger.info(f"Starting training {margin}"
                        f"max_epocs : {training_cfg.training.max_epochs}"
                        f"learning rate : {training_cfg.training.learning_rate}")
            
            save_path = f"{training_cfg.paths.save_model}/model_margin_{margin}.pt"
            train_model(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                val_loader=val_loader,
                epochs=training_cfg.training.max_epochs,
                lr=training_cfg.training.learning_rate,
                margin=margin,
                save_path=save_path
            )
            
        logger.success("Training completed")
        logger.success(f"Pipeline run complete. MLflow run ID: {run.info.run_id}")

if __name__ == "__main__":
    main()