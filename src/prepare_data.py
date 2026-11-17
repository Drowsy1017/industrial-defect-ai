
from pathlib import Path
import json

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

# 1. 專案路徑
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw"
REPORT_DIR = ROOT / "reports"
REPORT_DIR.mkdir(exist_ok=True)

# 2. 設定參數
BATCH_SIZE = 16
IMAGE_SIZE = 224
RANDOM_SEED = 42

# 3. 圖片前處理
# ImageNet 預訓練模型使用的正規化參數
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# 4. 先讀取圖片路徑及分類標籤
base_dataset = datasets.ImageFolder(DATA_DIR)

print("Class names:", base_dataset.classes)
print("Total images:", len(base_dataset))

if len(base_dataset) != 1800:
    raise ValueError("資料數量不是 1800，請檢查資料集")

# 5. 準備索引及標籤
indices = list(range(len(base_dataset)))
labels = base_dataset.targets

# 6. 分成訓練集 70% 和其餘 30%
train_idx, temp_idx = train_test_split(
    indices,
    test_size=0.30,
    random_state=RANDOM_SEED,
    stratify=labels,
)

# 7. 其餘 30% 平分成驗證集與測試集
temp_labels = [labels[i] for i in temp_idx]

val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=0.50,
    random_state=RANDOM_SEED,
    stratify=temp_labels,
)

# 8. 套用圖片轉換
image_dataset = datasets.ImageFolder(
    DATA_DIR,
    transform=transform,
)

train_dataset = Subset(image_dataset, train_idx)
val_dataset = Subset(image_dataset, val_idx)
test_dataset = Subset(image_dataset, test_idx)

# 9. 建立 DataLoader
# Windows 初學階段先使用 num_workers=0
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

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

# 10. 保存切分資訊
split_info = {
    "classes": base_dataset.classes,
    "train": train_idx,
    "val": val_idx,
    "test": test_idx,
}

with open(
    REPORT_DIR / "data_split.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(split_info, f, indent=2)

# 11. 顯示切分結果
print("\nDataset split:")
print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))
print("Test:", len(test_dataset))

# 12. 測試一個 Batch
images, batch_labels = next(iter(train_loader))

print("\nBatch shape:", images.shape)
print("Labels shape:", batch_labels.shape)

# 13. 測試 GPU 資料傳輸
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

images = images.to(device)

print("Device:", images.device)
print("\nData preparation completed!")
