
from pathlib import Path
import json
import csv

import torch
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
REPORT_DIR = ROOT / "reports"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# 1. 讀取原本的資料切分
with open(
    REPORT_DIR / "data_split.json",
    "r",
    encoding="utf-8"
) as f:
    split_info = json.load(f)

# 2. 使用與訓練相同的前處理
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

# 注意：本課只評估 Validation Set
val_indices = split_info["val"]
val_dataset = Subset(dataset, val_indices)

val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0,
)

# 3. 建立相同的模型結構
model = models.resnet18(weights=None)

model.fc = nn.Linear(
    model.fc.in_features,
    len(dataset.classes),
)

# 4. 載入訓練好的權重
checkpoint = torch.load(
    ROOT / "models" / "best_resnet18.pth",
    map_location=device,
    weights_only=True,
)

assert checkpoint["classes"] == dataset.classes

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)
model.eval()

print("Loaded model from epoch:", checkpoint["epoch"])

# 5. 執行驗證集推論
all_true = []
all_pred = []
all_confidence = []

with torch.inference_mode():
    for images, labels in val_loader:
        images = images.to(device)

        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)

        confidence, predictions = probabilities.max(dim=1)

        all_true.extend(labels.tolist())
        all_pred.extend(predictions.cpu().tolist())
        all_confidence.extend(confidence.cpu().tolist())

# 6. 評估指標
class_ids = list(range(len(dataset.classes)))

accuracy = accuracy_score(all_true, all_pred)

macro_f1 = f1_score(
    all_true,
    all_pred,
    labels=class_ids,
    average="macro",
    zero_division=0,
)

print(f"\nValidation Accuracy: {accuracy:.4f}")
print(f"Validation Macro F1: {macro_f1:.4f}")

print("\nClassification Report:")
print(classification_report(
    all_true,
    all_pred,
    labels=class_ids,
    target_names=dataset.classes,
    zero_division=0,
))

# 7. 產生混淆矩陣
cm = confusion_matrix(
    all_true,
    all_pred,
    labels=class_ids,
)

fig, ax = plt.subplots(figsize=(10, 8))

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=dataset.classes,
)

disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)

plt.title("Validation Confusion Matrix")
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "confusion_matrix.png",
    dpi=200,
)

plt.close()

# 8. 儲存錯誤分類清單
errors = []

for position, original_idx in enumerate(val_indices):
    true_label = all_true[position]
    pred_label = all_pred[position]

    if true_label != pred_label:
        path, _ = dataset.samples[original_idx]

        errors.append({
            "filename": str(Path(path).relative_to(DATA_DIR)),
            "true_class": dataset.classes[true_label],
            "predicted_class": dataset.classes[pred_label],
            "confidence": round(
                all_confidence[position], 4
            ),
        })

with open(
    REPORT_DIR / "misclassified.csv",
    "w",
    newline="",
    encoding="utf-8-sig",
) as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "filename",
            "true_class",
            "predicted_class",
            "confidence",
        ],
    )

    writer.writeheader()
    writer.writerows(errors)

print(f"\nMisclassified images: {len(errors)}")

for error in errors:
    print(error)

# 9. 繪製訓練 Loss 曲線
with open(
    REPORT_DIR / "training_history.json",
    "r",
    encoding="utf-8"
) as f:
    history = json.load(f)

epochs = [item["epoch"] for item in history]
losses = [item["train_loss"] for item in history]

plt.figure(figsize=(8, 5))
plt.plot(epochs, losses, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("ResNet18 Training Loss")
plt.xticks(epochs)
plt.grid(True)
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "training_loss.png",
    dpi=200,
)

plt.close()

print("\nEvaluation completed!")
