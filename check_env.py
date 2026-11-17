
import sys
import numpy as np
import pandas as pd
from PIL import Image
import sklearn
import matplotlib

print("=" * 40)
print("AI Industrial Defect Project")
print("=" * 40)

print(f"Python version: {sys.version.split()[0]}")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")

# 建立測試矩陣
data = np.array([
    [10, 20, 30],
    [40, 50, 60],
])

print("\nTest array:")
print(data)
print(f"Array shape: {data.shape}")

# 建立測試影像
image = Image.new("RGB", (200, 200), color="gray")
image.save("test_image.png")

print("\nImage created: test_image.png")
print("Environment check completed!")
