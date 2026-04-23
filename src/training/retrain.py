from pathlib import Path
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import sys
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.models.siamese_network import SiameseNetwork
from src.utils.image_utils import resize_with_padding
from src.utils.config_loader import retraining_cfg
from src.evaluation.contrastive_loss import ContrastiveLoss
from src.utils.split import writer_disjoint_split
from src.training.trainer import train_model

class FeedbackDataset(Dataset):
    def __init__(self, df, root_dir):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir

    def __len__(self):
        return len(self.df)

    def _load_image(self, rel_path):
        path = self.root_dir / Path(rel_path.replace("\\", "/"))

        img = Image.open(path).convert("L")
        img_np = np.array(img, dtype=np.uint8)

        img_np = resize_with_padding(img_np, target_size=256)
        img_np = img_np.astype(np.float32) / 255.0

        tensor = torch.from_numpy(img_np).unsqueeze(0)
        return tensor

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img1 = self._load_image(row["image1"])
        img2 = self._load_image(row["image2"])
        label = torch.tensor(float(row["label"]), dtype=torch.float32)

        return img1, img2, label

def retrain(model_path: Path, feedback_dir: Path):
    print("[INFO] Retraining started...")

    csv_path = feedback_dir / "feedback_pairs.csv"

    if not csv_path.exists():
        print("[WARN] No feedback data found. Skipping retraining.")
        return

    df = pd.read_csv(csv_path)

    if len(df) < training_cfg.training.df_length:
        print(f"[WARN] Not enough data ({len(df)} samples). Skipping retraining.")
        return

    train_df, val_df, test_df = writer_disjoint_split(df)

    print(f"[INFO] Split sizes ? Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    if len(val_df) == 0 or len(test_df) == 0:
        print("[WARN] Not enough writers for proper split. Skipping retraining.")
        return

    batch_size=retraining_cfg.training.batch_size

    train_dataset = FeedbackDataset(train_df, feedback_dir.parent)
    val_dataset   = FeedbackDataset(val_df, feedback_dir.parent)
    test_dataset  = FeedbackDataset(test_df, feedback_dir.parent)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SiameseNetwork(
        embedding_size=training_cfg.model.embedding_size
    )

    state_dict = torch.load(str(model_path), map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)

    # Freeze CNN
    for param in model.cnn.parameters():
        param.requires_grad = False

    model.cnn.eval()
    model.fc.train()

    timestamp = int(time.time())
    new_model_path = model_path.parent / f"model_v{timestamp}.pt"

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        epochs=retraining_cfg.training.max_epochs,
        lr=retraining_cfg.training.learning_rate,
        margin=retraining_cfg.loss.margin,
        save_path=new_model_path,
        patience=retraining_cfg.training.patience,
        min_delta=retraining_cfg.training.min_delta
    )

