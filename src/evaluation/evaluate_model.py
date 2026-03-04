import torch

@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0

    for img1, img2, label in dataloader:
        img1 = img1.to(device)
        img2 = img2.to(device)
        label = label.to(device)

        emb1, emb2 = model(img1, img2)
        loss = criterion(emb1, emb2, label)
        total_loss += loss.item()

    return total_loss / len(dataloader)