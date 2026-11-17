
from pathlib import Path
import json
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from sklearn.metrics import f1_score

# ==========================================
# 1. 設定參數
# ==========================================

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
SPLIT_FILE = ROOT / "reports" / "data_split.json"
MODEL_DIR = ROOT / "models"

MODEL_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.001
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")

# ==========================================
# 2. 讀取固定資料切分
# ==========================================

with open(SPLIT_FILE, "r", encoding="utf-8") as f:
    split_info = json.load(f)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

dataset = datasets.ImageFolder(
    DATA_DIR,
    transform=transform,
)

assert dataset.classes == split_info["classes"]
assert len(dataset) == 1800

train_dataset = Subset(dataset, split_info["train"])
val_dataset = Subset(dataset, split_info["val"])

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

# ==========================================
# 3. 建立 ResNet18 模型
# ==========================================

weights = models.ResNet18_Weights.DEFAULT

model = models.resnet18(weights=weights)

# 凍結預訓練特徵提取層
for parameter in model.parameters():
    parameter.requires_grad = False

# 更換最後的分類層
num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    len(dataset.classes),
)

model = model.to(device)

# ==========================================
# 4. Loss 與 Optimizer
# ==========================================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.fc.parameters(),
    lr=LEARNING_RATE,
)

best_f1 = -1.0
history = []

# ==========================================
# 5. 開始訓練
# ==========================================

for epoch in range(EPOCHS):

    # 特徵提取層保持 evaluation mode
    # 本次只訓練最後的分類層
    model.eval()
    model.fc.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward Pass
        outputs = model(images)

        # 計算損失
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # 更新權重
        optimizer.step()

        running_loss += loss.item() * labels.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = correct / total

    # ======================================
    # 6. 驗證模型
    # ======================================

    model.eval()

    val_correct = 0
    val_total = 0

    all_predictions = []
    all_labels = []

    with torch.inference_mode():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)

            all_predictions.extend(
                predictions.cpu().tolist()
            )
            all_labels.extend(
                labels.cpu().tolist()
            )

    val_acc = val_correct / val_total

    val_f1 = f1_score(
        all_labels,
        all_predictions,
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

    # 儲存驗證 F1 最佳的模型
    if val_f1 > best_f1:

        best_f1 = val_f1

        torch.save({
            "model_state_dict": model.state_dict(),
            "classes": dataset.classes,
            "epoch": epoch + 1,
            "val_macro_f1": val_f1,
        }, MODEL_DIR / "best_resnet18.pth")

        print("  -> Best model saved!")

# ==========================================
# 7. 儲存訓練紀錄
# ==========================================

with open(
    ROOT / "reports" / "training_history.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(history, f, indent=2)

print("\nTraining completed!")
print(f"Best validation Macro F1: {best_f1:.4f}")
print("Model: models/best_resnet18.pth")
