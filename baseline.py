
from ultralytics import YOLO

# 1. 사전학습된 YOLOv8n 모델 불러오기
model = YOLO("yolov8n.pt")

# 2. 샘플 이미지로 객체 탐지 테스트
results = model.predict(
    source="https://ultralytics.com/images/bus.jpg",
    save=True,
    imgsz=640
)

# 3. 결과 출력
for result in results:
    print("검출된 객체 수:", len(result.boxes))
    print("검출 결과:", result.boxes)

print("YOLOv8n Baseline 실행 완료!")
