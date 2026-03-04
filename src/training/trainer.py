import torch
import mlflow
from loguru import logger
from evaluation.contrastive_loss import ContrastiveLoss
from src.evaluation.evaluate_model import evaluate
from src.evaluation.evaluate_result import evaluate_classification
from src.evaluation.metrics import log_confusion_matrix

def train_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    epochs,
    lr,
    margin,
    save_path,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for param in model.cnn.parameters():
        param.requires_grad = False
    model.cnn.eval()
    logger.info("CNN backbone frozen")

    criterion = ContrastiveLoss(margin=margin)

    optimizer = torch.optim.AdamW(
        model.fc.parameters(),
        lr=float(lr)
    )

    mlflow.log_params({
        "epochs": epochs,
        "learning_rate": lr,
        "margin": margin,
        "optimizer": "AdamW",
        "cnn_frozen": True,
        "trainable_head": "fc",
        "batch_size": train_loader.batch_size,
        "device": device.type,
    })

    logger.info({
        "epochs": epochs,
        "learning_rate": lr,
        "margin": margin,
        "optimizer": "AdamW",
        "cnn_frozen": True,
        "trainable_head": "fc",
        "batch_size": train_loader.batch_size,
        "device": device.type,
    })

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.fc.train()
        total_train_loss = 0.0
        logger.info(f"Epoch: {epoch}")
        for img1, img2, label in train_loader:
            img1 = img1.to(device)
            img2 = img2.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            emb1, emb2 = model(img1, img2)
            loss = criterion(emb1, emb2, label)
            logger.info(f"Loss is: {loss}")
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        train_loss = total_train_loss / len(train_loader)
        logger.info(f"Average loss for epoch {epoch}: {train_loss}")
        
        val_loss = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device
        )

        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)

        logger.info(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            mlflow.log_artifact(save_path, artifact_path="best_model")
            logger.success("✅ Best model updated")

    logger.info("Loading best validation model for testing")
    model.load_state_dict(torch.load(save_path, map_location=device))
    model.to(device)

    test_loss = evaluate(
        model=model,
        dataloader=test_loader,
        criterion=criterion,
        device=device
    )

    mlflow.log_metric("test_loss", test_loss)
    logger.success(f"📊 Final Test Loss: {test_loss:.4f}")

    metrics = evaluate_classification(
    model=model,
    dataloader=test_loader,
    device=device,
    threshold=margin)

    mlflow.log_metric("test_accuracy", metrics["accuracy"])
    mlflow.log_metric("test_precision", metrics["precision"])
    mlflow.log_metric("test_recall", metrics["recall"])
    mlflow.log_metric("test_f1_score", metrics["f1_score"])
    cm = metrics["confusion_matrix"]
    logger.info(f"Confusion Matrix:\n{cm}")

    log_confusion_matrix(cm)





