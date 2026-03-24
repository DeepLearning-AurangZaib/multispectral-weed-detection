# ================================
# Minimal SegFormer Training
# ================================

import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torch import nn, optim
from transformers import SegformerForSemanticSegmentation

# -------- CONFIG --------
NUM_CLASSES = 3
BATCH_SIZE = 8
EPOCHS = 5
LR = 5e-5
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

        mask = Image.open(
            os.path.join(self.mask_dir, name.rsplit(".",1)[0] + ".png")
        ).convert("L").resize((IMAGE_SIZE, IMAGE_SIZE))

        mask = torch.from_numpy(np.array(mask)).long()

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

model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512",
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)

# -------- TRAIN --------
print("🚀 Training started")

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    for imgs, masks in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)

        outputs = model(imgs).logits

        masks_resized = torch.nn.functional.interpolate(
            masks.unsqueeze(1).float(),
            size=outputs.shape[2:],
            mode="nearest"
        ).squeeze(1).long()

        loss = criterion(outputs, masks_resized)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # -------- VALIDATION --------
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)

            outputs = model(imgs).logits
            preds = torch.argmax(outputs, dim=1)

            masks_resized = torch.nn.functional.interpolate(
                masks.unsqueeze(1).float(),
                size=preds.shape[1:],
                mode="nearest"
            ).squeeze(1).long()

            correct += (preds == masks_resized).sum().item()
            total += masks_resized.numel()

    acc = 100 * correct / total

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss:.4f} | Val Acc: {acc:.2f}%")

print("✅ Training complete")