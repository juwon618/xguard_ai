
from ultralytics import YOLO
import torch

MODEL_PATH = (
    "runs/detect/runs/xguard/"
    "zenodo_v2/weights/best.pt"
)

DATA_PATH = "dataset/yolo_zenodo_v2/data.yaml"

device = "mps" if torch.backends.mps.is_available() else "cpu"

model = YOLO(MODEL_PATH)

results = model.val(
    data=DATA_PATH,
    split="test",
    imgsz=256,
    batch=4,
    device=device,
    workers=0,
    plots=True,
    project="runs/xguard",
    name="zenodo_v2_test"
)

print("\n===== Test 성능 =====")
print("Precision:", results.box.mp)
print("Recall:", results.box.mr)
print("mAP50:", results.box.map50)
print("mAP50-95:", results.box.map)
