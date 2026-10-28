
from ultralytics import YOLO
from pathlib import Path

# 데이터셋 설정
DATASET_CONFIG = Path("dataset/yolo_gdxray/data.yaml")

# 사전학습된 YOLOv8n 불러오기
model = YOLO("yolov8n.pt")

# 데이터셋이 준비된 경우에만 학습
if DATASET_CONFIG.exists():
    results = model.train(
    data=str(DATASET_CONFIG),
    epochs=3,
    imgsz=640,
    batch=4,
    device="mps",
    project="runs/xguard",
    name="baseline_test"
    )
    
    print("학습 완료!")
else:
    print("데이터셋 미준비: 학습 코드 설정 완료")
