
from pathlib import Path
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

# 1. 경로 설정
MODEL_PATH = (
    "runs/detect/runs/xguard/"
    "zenodo_baseline/weights/best.pt"
)

ROOT = Path("dataset/yolo_zenodo")
IMAGE_DIR = ROOT / "images/val"
LABEL_DIR = ROOT / "labels/val"

# 2. 학습된 모델 불러오기
model = YOLO(MODEL_PATH)

# 3. Validation 이미지 4장 선택
random.seed(42)
images = random.sample(
    sorted(IMAGE_DIR.glob("*.png")), 4
)

# 4. 모델 예측
results = model.predict(
    source=[str(p) for p in images],
    imgsz=256,
    conf=0.25,
    device="mps",
    verbose=False
)

# 5. GT와 AI 예측 비교
fig, axes = plt.subplots(4, 2, figsize=(9, 16))

for i, (image_path, result) in enumerate(zip(images, results)):
    img = Image.open(image_path).convert("RGB")
    width, height = img.size

    # 왼쪽: Ground Truth
    ax_gt = axes[i, 0]
    ax_gt.imshow(img)
    ax_gt.set_title(f"GT: {image_path.stem}", fontsize=9)

    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    for line in label_path.read_text().splitlines():
        cls, cx, cy, w, h = map(float, line.split())

        x = (cx - w / 2) * width
        y = (cy - h / 2) * height

        ax_gt.add_patch(
            patches.Rectangle(
                (x, y),
                w * width,
                h * height,
                fill=False,
                edgecolor="lime",
                linewidth=1.5
            )
        )

    # 오른쪽: AI 예측
    ax_pred = axes[i, 1]
    ax_pred.imshow(img)
    ax_pred.set_title(
        f"Prediction: {len(result.boxes)} boxes",
        fontsize=10
    )

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        confidence = float(box.conf[0])

        ax_pred.add_patch(
            patches.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor="red",
                linewidth=1.5
            )
        )

        ax_pred.text(
            x1,
            max(0, y1 - 3),
            f"{confidence:.2f}",
            color="red",
            fontsize=8
        )

    ax_gt.axis("off")
    ax_pred.axis("off")

plt.tight_layout()

# 결과 이미지 저장
output = Path("runs/zenodo_prediction_compare.png")
output.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output, dpi=150)
plt.show()

print("결과 저장:", output)
