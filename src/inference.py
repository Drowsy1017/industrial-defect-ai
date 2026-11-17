
from pathlib import Path
from time import perf_counter

import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "best_resnet18_finetuned.pth"


class DefectClassifier:
    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        checkpoint = torch.load(
            MODEL_PATH,
            map_location="cpu",
            weights_only=True,
        )

        self.classes = checkpoint["classes"]
        self.epoch = checkpoint["epoch"]

        self.model = models.resnet18(weights=None)

        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            len(self.classes),
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model = self.model.to(self.device)
        self.model.eval()

        # 必須與模型訓練時的前處理一致
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        print(f"Model loaded on {self.device}")

    def predict(self, image: Image.Image):
        start = perf_counter()

        image = image.convert("RGB")

        tensor = self.transform(image)
        tensor = tensor.unsqueeze(0).to(self.device)

        with torch.inference_mode():
            outputs = self.model(tensor)
            probabilities = torch.softmax(outputs, dim=1)

        scores = probabilities[0].cpu().tolist()

        predicted_idx = int(
            torch.argmax(probabilities, dim=1).item()
        )

        # 等待 GPU 完成，避免低估推論時間
        if self.device.type == "cuda":
            torch.cuda.synchronize()

        elapsed_ms = (perf_counter() - start) * 1000

        return {
            "predicted_class": self.classes[predicted_idx],
            "confidence": round(scores[predicted_idx], 6),
            "probabilities": {
                name: round(float(score), 6)
                for name, score in zip(self.classes, scores)
            },
            "inference_time_ms": round(elapsed_ms, 2),
            "device": str(self.device),
            "model_epoch": self.epoch,
        }
