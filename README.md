
# X-Guard AIoT — YOLOv8n Object Detection

oneM2M & ETSI MEC 기반 AIoT 이물질 자동 검사 시스템의 AI 객체 탐지 모듈입니다.

## 1. 프로젝트 개요

- **모델:** YOLOv8n
- **데이터셋:** [Zenodo X-ray Object Detection Dataset](https://zenodo.org/records/5681008)
- **탐지 대상:** 점토 내부의 돌(Stone)
- **학습 환경:** Python 3.11, PyTorch, Apple M1 MPS
- **학습:** 20 Epoch, 이미지 크기 256×256

## 2. 최종 Test 성능

| Metric | Result |
|---|---:|
| Precision | 92.04% |
| Recall | 85.42% |
| mAP@50 | 91.19% |
| mAP@50–95 | 56.92% |

독립 Test: 280장, GT 바운딩박스 590개.

위 수치는 Zenodo X-ray 데이터셋에서 측정한 결과이며, 실제 근적외선 카메라 영상에 대한 성능은 별도 검증이 필요합니다.

## 3. 설치

```bash
pip install -r requirements.txt
```

## 4. 모델 사용

학습된 모델은 `models/xguard_yolov8n_best.pt`에 있습니다.

```python
from ultralytics import YOLO

model = YOLO("models/xguard_yolov8n_best.pt")
results = model.predict("test_image.png", conf=0.25)

for result in results:
    print(result.boxes)
```

## 5. 주요 파일

- `prepare_zenodo_v2.py`: 데이터셋 준비 및 YOLO 라벨 변환
- `train_zenodo_v2.py`: YOLOv8n 학습
- `test_zenodo_v2.py`: 독립 Test 평가
- `predict_zenodo.py`: 예측 결과 시각화
- `models/xguard_yolov8n_best.pt`: 학습된 모델 가중치

데이터셋은 GitHub에 포함하지 않습니다. 학습 및 평가를 재현하려면 Zenodo 원본 데이터셋을 별도로 다운로드해야 합니다.

## 6. X-Guard 연동 계획

이미지 입력 → YOLOv8n 객체 탐지 → PASS/DEFECT/WARNING 판정 → oneM2M 결과 저장 → 자동 리젝터 제어

현재 모델은 X-ray 기반 Baseline이며, 실제 카메라 연동 및 MEC 환경 검증은 후속 작업입니다.
