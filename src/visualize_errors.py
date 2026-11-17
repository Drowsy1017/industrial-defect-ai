
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data" / "raw"
REPORT_DIR = ROOT / "reports"

# 讀取模型分類錯誤紀錄
df = pd.read_csv(
    REPORT_DIR / "misclassified.csv"
)

print(f"Misclassified images: {len(df)}")

if df.empty:
    print("No misclassified images found.")
    raise SystemExit(0)

# 每行顯示一張錯誤圖片
fig, axes = plt.subplots(
    len(df),
    1,
    figsize=(8, 4 * len(df)),
    squeeze=False,
)

for i, row in df.iterrows():

    image_path = DATA_DIR / row["filename"]

    with Image.open(image_path) as img:
        image = img.convert("L").copy()

    ax = axes[i, 0]

    ax.imshow(image, cmap="gray")
    ax.axis("off")

    ax.set_title(
        f"True: {row['true_class']}\n"
        f"Predicted: {row['predicted_class']}\n"
        f"Confidence: {row['confidence']:.2%}"
    )

plt.tight_layout()

output_path = REPORT_DIR / "error_analysis.png"

plt.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print(f"Saved: {output_path}")
