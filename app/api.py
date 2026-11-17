
from contextlib import asynccontextmanager
from io import BytesIO
import hashlib

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
    Query,
)
from fastapi.concurrency import run_in_threadpool
from PIL import Image, UnidentifiedImageError

from app.inference import DefectClassifier
from app.database import (
    init_database,
    save_prediction,
    get_history,
    get_statistics,
)

classifier = None

MAX_FILE_SIZE = 10 * 1024 * 1024

ALLOWED_TYPES = {
    "image/png",
    "image/jpeg",
    "image/bmp",
    "image/x-ms-bmp",
}


# ==========================================
# 1. 初始化服務
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier

    init_database()
    classifier = DefectClassifier()

    print("Database initialized")
    print("AI model initialized")

    yield

    classifier = None


app = FastAPI(
    title="Industrial Defect AI API",
    description="ResNet18 steel defect classification",
    version="1.1.0",
    lifespan=lifespan,
)


# ==========================================
# 2. 健康檢查
# ==========================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": classifier is not None,
        "device": (
            str(classifier.device)
            if classifier
            else None
        ),
        "database": "sqlite",
    }


# ==========================================
# 3. AI 辨識與歷史紀錄
# ==========================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded",
        )

    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Unsupported image type",
        )

    try:
        content = await file.read(MAX_FILE_SIZE + 1)

        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="File too large",
            )

        image_sha256 = hashlib.sha256(
            content
        ).hexdigest()

        with Image.open(BytesIO(content)) as img:
            img.load()
            image = img.convert("RGB")

    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Invalid image file",
        )

    finally:
        await file.close()

    # 在工作執行緒中處理同步推論與資料庫寫入
    def process_prediction():
        result = classifier.predict(image)

        record_id = save_prediction(
            filename=file.filename or "unknown",
            image_sha256=image_sha256,
            result=result,
        )

        return {
            "record_id": record_id,
            "filename": file.filename,
            **result,
        }

    return await run_in_threadpool(
        process_prediction
    )


# ==========================================
# 4. 查詢歷史紀錄
# ==========================================

@app.get("/history")
def history(
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    records = get_history(
        limit=limit,
        offset=offset,
    )

    return {
        "count": len(records),
        "records": records,
    }


# ==========================================
# 5. 統計資料
# ==========================================

@app.get("/statistics")
def statistics():
    return get_statistics()
