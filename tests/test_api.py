
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image
from fastapi.testclient import TestClient

from app import api, database


# =====================================
# 1. 模擬 AI 模型
# =====================================

class FakeClassifier:

    def __init__(self):
        self.device = "cpu"

    def predict(self, image):
        return {
            "predicted_class": "scratches",
            "confidence": 0.99,
            "probabilities": {
                "crazing": 0.002,
                "inclusion": 0.002,
                "patches": 0.002,
                "pitted_surface": 0.002,
                "rolled_in_scale": 0.002,
                "scratches": 0.99,
            },
            "inference_time_ms": 1.5,
            "device": "cpu",
            "model_epoch": 4,
        }


# =====================================
# 2. 建立測試圖片
# =====================================

def create_test_image():
    image = Image.new(
        "RGB",
        (100, 100),
        color="gray",
    )

    buffer = BytesIO()
    image.save(buffer, format="PNG")

    return buffer.getvalue()


# =====================================
# 3. 建立獨立測試環境
# =====================================

@pytest.fixture
def client(tmp_path, monkeypatch):

    test_db = tmp_path / "test_predictions.db"

    # 將資料庫位置切換到測試用檔案
    monkeypatch.setattr(
        database,
        "DB_PATH",
        test_db,
    )

    # 將真實 ResNet18 替換為模擬模型
    monkeypatch.setattr(
        api,
        "DefectClassifier",
        FakeClassifier,
    )

    # 啟動 FastAPI 測試客戶端
    with TestClient(api.app) as test_client:
        yield test_client


# =====================================
# Test 1：健康檢查
# =====================================

def test_health(client):

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["database"] == "sqlite"


# =====================================
# Test 2：正常圖片辨識
# =====================================

def test_predict_success(client):

    image_bytes = create_test_image()

    response = client.post(
        "/predict",
        files={
            "file": (
                "test.png",
                image_bytes,
                "image/png",
            )
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["predicted_class"] == "scratches"
    assert data["confidence"] == 0.99
    assert isinstance(data["record_id"], int)

    history = client.get("/history").json()

    assert history["count"] == 1
    assert history["records"][0]["filename"] == "test.png"


# =====================================
# Test 3：不支援的檔案格式
# =====================================

def test_unsupported_file(client):

    response = client.post(
        "/predict",
        files={
            "file": (
                "test.txt",
                b"hello",
                "text/plain",
            )
        },
    )

    assert response.status_code == 415


# =====================================
# Test 4：損壞的圖片
# =====================================

def test_invalid_image(client):

    response = client.post(
        "/predict",
        files={
            "file": (
                "broken.png",
                b"this is not a real image",
                "image/png",
            )
        },
    )

    assert response.status_code == 400


# =====================================
# Test 5：檔案超過 10 MB
# =====================================

def test_file_too_large(client):

    oversized = b"x" * (
        api.MAX_FILE_SIZE + 1
    )

    response = client.post(
        "/predict",
        files={
            "file": (
                "large.png",
                oversized,
                "image/png",
            )
        },
    )

    assert response.status_code == 413


# =====================================
# Test 6：統計資料
# =====================================

def test_statistics(client):

    image_bytes = create_test_image()

    response = client.post(
        "/predict",
        files={
            "file": (
                "test.png",
                image_bytes,
                "image/png",
            )
        },
    )

    assert response.status_code == 200

    response = client.get("/statistics")

    assert response.status_code == 200

    data = response.json()

    assert data["total_predictions"] == 1
    assert (
        data["class_distribution"][0]["predicted_class"]
        == "scratches"
    )
    assert data["average_inference_ms"] == 1.5


# =====================================
# Test 7：測試資料庫隔離
# =====================================

def test_history_starts_empty(client):

    response = client.get("/history")

    assert response.status_code == 200
    assert response.json()["records"] == []
