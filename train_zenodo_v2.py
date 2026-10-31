
from ultralytics import YOLO
import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"

print("사용 장치:", device)

# 동일한 사전학습 모델에서 새롭게 시작
model = YOLO("yolov8n.pt")

model.train(
    data="dataset/yolo_zenodo_v2/data.yaml",
    epochs=20,
    imgsz=256,
    batch=4,
    device=device,
    workers=0,

    # X-ray 흑백 영상이므로 색상 증강 비활성화
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.0,

    project="runs/xguard",
    name="zenodo_v2",
    plots=True
)

print("Zenodo v2 학습 완료!")
