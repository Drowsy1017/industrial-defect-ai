
from pathlib import Path
import json
import csv

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")

# 1. 讀取既有的資料切分
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
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

dataset = datasets.ImageFolder(
    DATA_DIR,
    transform=transform,
)

assert dataset.classes == split["classes"]
assert len(dataset) == 1800

# 確認三組索引沒有重疊
train_ids = set(split["train"])
val_ids = set(split["val"])
test_ids = set(split["test"])

assert not (train_ids & val_ids)
assert not (train_ids & test_ids)
assert not (val_ids & test_ids)

test_dataset = Subset(
    dataset,
    split["test"],
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0,
)

# 2. 比較的模型
model_files = {
    "Baseline": "best_resnet18.pth",
    "Fine-tuning": "best_resnet18_finetuned.pth",
}

results = []

# 3. 依序測試兩個模型
for model_name, filename in model_files.items():

    print(f"\n{'=' * 45}")
    print(f"Evaluating: {model_name}")
    print(f"{'=' * 45}")

    model = models.resnet18(weights=None)

    model.fc = nn.Linear(
        model.fc.in_features,
        len(dataset.classes),
    )

    checkpoint = torch.load(
        MODEL_DIR / filename,
        map_location="cpu",
        weights_only=True,
    )

    assert checkpoint["classes"] == dataset.classes

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)
    model.eval()

    all_true = []
    all_pred = []

    with torch.inference_mode():

        for images, labels in test_loader:

            images = images.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            all_true.extend(labels.tolist())
            all_pred.extend(
                predictions.cpu().tolist()
            )

    # 4. 計算測試指標
    class_ids = list(range(len(dataset.classes)))

    accuracy = accuracy_score(
        all_true, all_pred
    )

    macro_f1 = f1_score(
        all_true,
        all_pred,
        labels=class_ids,
        average="macro",
        zero_division=0,
    )

    errors = sum(
        t != p
        for t, p in zip(all_true, all_pred)
    )

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Macro F1: {macro_f1:.4f}")
    print(f"Misclassified: {errors}")

    print("\nClassification Report:")

    print(classification_report(
        all_true,
        all_pred,
        labels=class_ids,
        target_names=dataset.classes,
        zero_division=0,
    ))

    
    # 計算混淆矩陣
    cm = confusion_matrix(
        all_true,
        all_pred,
        labels=class_ids,
    )

    print("Confusion Matrix:")
    print(cm)

    # 繪製混淆矩陣
    fig, ax = plt.subplots(figsize=(10, 8))

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=dataset.classes,
    )

    disp.plot(
        ax=ax,
        cmap="Blues",
        xticks_rotation=45,
        values_format="d",
    )

    ax.set_title(
        f"{model_name} - Test Confusion Matrix"
    )

    plt.tight_layout()

    # 依照模型名稱儲存
    filename = (
        "baseline_test_confusion_matrix.png"
        if model_name == "Baseline"
        else "finetuned_test_confusion_matrix.png"
    )

    output_path = REPORT_DIR / filename

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved confusion matrix: {output_path}")


    results.append({
        "model": model_name,
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "errors": errors,
        "samples": len(all_true),
    })

# 5. 儲存比較結果
output_path = REPORT_DIR / "model_comparison.csv"

with open(
    output_path,
    "w",
    newline="",
    encoding="utf-8-sig",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "model",
            "accuracy",
            "macro_f1",
            "errors",
            "samples",
        ],
    )

    writer.writeheader()
    writer.writerows(results)

print("\nModel comparison completed!")
print(f"Saved to: {output_path}")
