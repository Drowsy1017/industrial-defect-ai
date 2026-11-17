
from pathlib import Path
import json
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(exist_ok=True)

SEED = 42
EPOCHS = 5
BATCH_SIZE = 16

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")

# 讀取相同的資料切分
with open(
    REPORT_DIR / "data_split.json",
    "r",
    encoding="utf-8",
) as f:
    split = json.load(f)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])

dataset = datasets.ImageFolder(
    DATA_DIR, transform=transform
)

assert dataset.classes == split["classes"]
assert len(dataset) == 1800

train_dataset = Subset(dataset, split["train"])
val_dataset = Subset(dataset, split["val"])

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

# 建立 ResNet18
model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)

# 先凍結所有層
for param in model.parameters():
    param.requires_grad = False

# 解凍 Layer4
for param in model.layer4.parameters():
    param.requires_grad = True

# 建立新的六類分類層
model.fc = nn.Linear(
    model.fc.in_features,
    len(dataset.classes),
)

model = model.to(device)

criterion = nn.CrossEntropyLoss()

# 不同模組使用不同 Learning Rate
optimizer = torch.optim.Adam([
    {
        "params": model.layer4.parameters(),
        "lr": 0.0001,
    },
    {
        "params": model.fc.parameters(),
        "lr": 0.001,
    },
])

best_f1 = -1.0
history = []

for epoch in range(EPOCHS):

    # 保持 BatchNorm running statistics 不變
    model.eval()

    # 只啟用需訓練的模組
    model.layer4.train()
    model.fc.train()

    # ResNet18 的 Layer4 內有 BatchNorm
    # 保留 ImageNet 預訓練時的 running statistics
    for module in model.layer4.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    train_loss = total_loss / total
    train_acc = correct / total

    # 驗證階段
    model.eval()

    true_labels = []
    pred_labels = []

    with torch.inference_mode():
        for images, labels in val_loader:

            images = images.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            true_labels.extend(labels.tolist())
            pred_labels.extend(
                predictions.cpu().tolist()
            )

    val_acc = sum(
        t == p
        for t, p in zip(true_labels, pred_labels)
    ) / len(true_labels)

    val_f1 = f1_score(
        true_labels,
        pred_labels,
        labels=list(range(len(dataset.classes))),
        average="macro",
        zero_division=0,
    )

    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "val_macro_f1": val_f1,
    })

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Acc: {val_acc:.4f} | "
        f"Val F1: {val_f1:.4f}"
    )

    if val_f1 > best_f1:

        best_f1 = val_f1

        torch.save({
            "model_state_dict": model.state_dict(),
            "classes": dataset.classes,
            "epoch": epoch + 1,
            "val_macro_f1": val_f1,
        }, MODEL_DIR / "best_resnet18_finetuned.pth")

        print("  -> Best fine-tuned model saved!")

with open(
    REPORT_DIR / "finetune_history.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(history, f, indent=2)

print("\nFine-tuning completed!")
print(f"Best validation Macro F1: {best_f1:.4f}")
