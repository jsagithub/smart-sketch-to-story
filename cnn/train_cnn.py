import json

import torch
import torch.nn as nn
from torch.optim import Adam
from model import SketchCNN
from dataset import TUBerlinParquet
from torch.utils.data import random_split, DataLoader
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

train_set = TUBerlinParquet("../data/datasets/train-00000-of-00001.parquet", transform=transform)
val_set   = TUBerlinParquet("../data/datasets/validation-00000-of-00001.parquet", transform=transform)
test_set  = TUBerlinParquet("../data/datasets/test-00000-of-00001.parquet", transform=transform)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=64, shuffle=False)
test_loader  = DataLoader(test_set,  batch_size=64, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"

# determine number of classes from the dataset (fallback to scanning if needed)
num_classes = None
if hasattr(train_set, "num_classes"):
    num_classes = train_set.num_classes
elif hasattr(train_set, "classes"):
    num_classes = len(train_set.classes)
elif hasattr(train_set, "labels"):
    labels = getattr(train_set, "labels")
    try:
        num_classes = int(max(labels)) + 1
    except Exception:
        num_classes = len(set(int(x) for x in labels))
else:
    labels_set = set()
    for _, lab in train_set:
        labels_set.add(int(lab))
    num_classes = len(labels_set)

model = SketchCNN(num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-3)

# 4. Training loop
epochs = 20

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # ===== validation =====
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss:.3f} - Val Acc: {acc:.3f}")

torch.save(model.state_dict(), "cnn_best.pt")
idx_to_class = {v: k for k, v in train_set.class_to_idx.items()}

with open("cnn/idx_to_class.json", "w") as f:
    json.dump(idx_to_class, f, indent=2)

print("Saved class mapping to cnn/idx_to_class.json")
print("Model saved!")