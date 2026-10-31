
from ultralytics import YOLO
from pathlib import Path
import torch

# 데이터셋 경로
DATASET = Path("dataset/yolo_zenodo/data.yaml")

# MacBook M1 GPU 사용
device = "mps" if torch.backends.mps.is_available() else "cpu"

print("사용 장치:", device)

# 사전학습 모델
model = YOLO("yolov8n.pt")

# Baseline 테스트 학습
model.train(
    data=str(DATASET),
    epochs=3,
    imgsz=256,
    batch=4,
    device=device,
    workers=0,
    project="runs/xguard",
    name="zenodo_baseline",
    plots=True
)

print("Zenodo Baseline 학습 완료!")
