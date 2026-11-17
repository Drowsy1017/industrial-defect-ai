FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=2
ENV MKL_NUM_THREADS=2

# 安裝 PyTorch CPU 版本
RUN python -m pip install --no-cache-dir \
    torch==2.14.1 torchvision==0.29.1 \
    --index-url https://download.pytorch.org/whl/cpu

# 安裝 Web API 與 Dashboard 套件
COPY requirements-docker.txt .

RUN python -m pip install --no-cache-dir \
    -r requirements-docker.txt

COPY app/ /app/app/

RUN mkdir -p /app/models /app/data/database

EXPOSE 8000 8501

CMD ["python", "-m", "uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
