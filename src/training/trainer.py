import torch
from evaluation.contrastive_loss import ContrastiveLoss
from loguru import logger



def train_model(model, train_loader, epochs, lr, margin, save_path):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)

    # Freeze CNN backbone
    for param in model.cnn.parameters():
        param.requires_grad = False
    logger.info("CNN Freezed")
    # Put CNN in eval mode (important for BatchNorm)
    model.cnn.eval()

    criterion = ContrastiveLoss(margin=margin)

    optimizer = torch.optim.AdamW(
        model.fc.parameters(),
        lr=float(lr)
    )
    logger.info("Started iteration")

    for epoch in range(epochs):

        model.fc.train()
        total_loss = 0

        for img1, img2, label in train_loader:

            img1 = img1.to(device)
            img2 = img2.to(device)
            label = label.to(device)
            logger.info(f"Image1 {img1}")
            logger.info(f"Image2 {img2}")
            logger.info(f"Label {label}")

            optimizer.zero_grad()

            emb1, emb2 = model(img1, img2)

            loss = criterion(emb1, emb2, label)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            logger.info(f"Total loss: {total_loss}")

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} Loss = {avg_loss:.6f}")

    torch.save(model.state_dict(), save_path)
    logger.info("✅ Model saved")