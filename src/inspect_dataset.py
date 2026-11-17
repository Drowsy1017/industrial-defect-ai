
from pathlib import Path
from PIL import Image

# 專案根目錄
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 圖片資料夾
DATA_DIR = PROJECT_ROOT / "data" / "raw"

# 六種瑕疵類別
CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled_in_scale",
    "scratches",
]

print("=" * 40)
print("NEU Dataset Inspection")
print("=" * 40)

total = 0

for class_name in CLASSES:
    folder = DATA_DIR / class_name

    if not folder.exists():
        print(f"[MISSING] {class_name}")
        continue

    images = [
        p for p in folder.iterdir()
        if p.is_file()
        and p.suffix.lower() in
        [".bmp", ".jpg", ".jpeg", ".png"]
    ]

    count = len(images)
    total += count

    print(f"{class_name}: {count} images")

print("-" * 40)
print(f"Total images: {total}")

if total == 1800:
    print("[SUCCESS] Dataset count is correct!")
else:
    print("[WARNING] Check your dataset folders.")
