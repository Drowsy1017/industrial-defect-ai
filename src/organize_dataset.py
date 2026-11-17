
from pathlib import Path
from collections import Counter
import re
import shutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SOURCE_DIR = PROJECT_ROOT / "data" / "downloaded"
TARGET_DIR = PROJECT_ROOT / "data" / "raw"

CLASS_MAP = {
    "Cr": "crazing",
    "In": "inclusion",
    "Pa": "patches",
    "PS": "pitted_surface",
    "RS": "rolled_in_scale",
    "Sc": "scratches",
}

IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png"}

# 預設只預覽，不真正複製
DRY_RUN = False

counts = Counter()
skipped = []

for image_path in SOURCE_DIR.rglob("*"):
    if not image_path.is_file():
        continue

    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
        continue

    # 例如 Cr_1.bmp、PS_12.bmp
    match = re.match(
        r"^(Cr|In|Pa|PS|RS|Sc)[_-]\d+$",
        image_path.stem,
        flags=re.IGNORECASE,
    )

    if not match:
        skipped.append(str(image_path))
        continue

    prefix = match.group(1)

    class_name = next(
        name for code, name in CLASS_MAP.items()
        if code.lower() == prefix.lower()
    )

    destination = TARGET_DIR / class_name / image_path.name

    if destination.exists():
        skipped.append(f"檔名衝突：{destination}")
        continue

    print(f"{image_path.name} -> {class_name}/")

    if not DRY_RUN:
        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        shutil.copy2(image_path, destination)

    counts[class_name] += 1

print("\n分類結果：")

for class_name in CLASS_MAP.values():
    print(f"{class_name}: {counts[class_name]}")

print(f"\n未分類或略過：{len(skipped)}")

for item in skipped[:10]:
    print(f"  {item}")

if DRY_RUN:
    print("\n目前為預覽模式，沒有複製圖片。")
