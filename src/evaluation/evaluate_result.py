import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)

@torch.no_grad()
def evaluate_classification(model, dataloader, device, threshold):
    model.eval()

    y_true = []
    y_pred = []

    for img1, img2, label in dataloader:
        img1 = img1.to(device)
        img2 = img2.to(device)
        label = label.to(device)

        emb1, emb2 = model(img1, img2)
        dist = torch.norm(emb1 - emb2, p=2, dim=1)

        preds = (dist < threshold).long()

        y_true.extend(label.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }

@torch.no_grad()
def find_best_threshold(model, dataloader, device, thresholds):
    best_thr = None
    best_f1 = -1

    for thr in thresholds:
        metrics = evaluate_classification(model, dataloader, device, thr)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_thr = thr

    return best_thr, best_f1