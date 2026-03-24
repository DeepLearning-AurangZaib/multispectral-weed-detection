# ================================
# Minimal DeepLabV3+ Training
# ================================

import os
import torch
import torchvision
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn, optim
import cv2

# -------- CONFIG --------
NUM_CLASSES = 3
BATCH_SIZE = 8
EPOCHS = 5
LR = 1e-4
IMAGE_SIZE = 640
VAL_SPLIT = 0.3

DATA_ROOT = "PATH_TO_DATA"
IMG_DIR = os.path.join(DATA_ROOT, "images")
MASK_DIR = os.path.join(DATA_ROOT, "masks")

# -------- DATASET --------
class DatasetSeg(Dataset):
    def __init__(self, img_dir, mask_dir):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(img_dir))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        name = self.images[idx]

        img = Image.open(os.path.join(self.img_dir, name)).resize((IMAGE_SIZE, IMAGE_SIZE))
        img = np.array(img).astype(np.float32)

        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)

        img = torch.from_numpy(img).permute(2, 0, 1)

        mask = cv2.imread(
            os.path.join(self.mask_dir, name.rsplit(".",1)[0] + ".png"),
            cv2.IMREAD_GRAYSCALE
        )
        mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
        mask = torch.from_numpy(mask).long()

        return img, mask

# -------- LOAD DATA --------
dataset = DatasetSeg(IMG_DIR, MASK_DIR)
val_len = int(len(dataset) * VAL_SPLIT)
train_len = len(dataset) - val_len

train_ds, val_ds = random_split(dataset, [train_len, val_len])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

# -------- MODEL --------
device = "cuda" if torch.cuda.is_available() else "cpu"

model = torchvision.models.segmentation.deeplabv3_mobilenet_v3_large(weights="DEFAULT")
model.classifier = torchvision.models.segmentation.deeplabv3.DeepLabHead(960, NUM_CLASSES)
model.aux_classifier = None
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)

# -------- TRAIN --------
print("🚀 Training started")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    for imgs, masks in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)["out"]
        loss = criterion(outputs, masks)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # -------- VALIDATION --------
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)

            outputs = model(imgs)["out"]
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == masks).sum().item()
            total += masks.numel()

    acc = 100 * correct / total

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss:.4f} | Val Acc: {acc:.2f}%")

print("✅ Training complete")