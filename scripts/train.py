import torch
import torch.nn as nn
import torch.optim as optim
from src.model.model import RockfallDeepLab
from src.data.dataloaders import get_loader
from src.training.metrics import compute_mIoU

IMG_DIR = "data/Images"
MASK_DIR = "data/Masks"
NUM_CLASSES = 6
EPOCHS = 50
BATCH_SIZE = 8
LR = 1e-4

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = RockfallDeepLab(num_classes=NUM_CLASSES).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    loader = get_loader(IMG_DIR, MASK_DIR, batch_size=BATCH_SIZE)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)["out"]
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(loader)}")

if __name__ == "__main__":
    train()
