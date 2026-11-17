
import torch
import torchvision

print("=" * 45)
print("PyTorch GPU Environment Check")
print("=" * 45)

print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")
print(f"PyTorch CUDA version: {torch.version.cuda}")

cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

if cuda_available:
    device = torch.device("cuda:0")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # 建立兩個隨機矩陣
    a = torch.rand(1000, 1000, device=device)
    b = torch.rand(1000, 1000, device=device)

    # 在 GPU 上執行矩陣乘法
    result = torch.matmul(a, b)
    torch.cuda.synchronize()

    print(f"Result shape: {result.shape}")
    print(f"Result device: {result.device}")
    print("GPU matrix multiplication successful!")
else:
    print("CUDA is unavailable. Check PyTorch installation.")
