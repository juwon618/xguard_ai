
from pathlib import Path
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

folder = Path(
    "dataset/croppedgdxray/"
    "cropped_castings_128_256/C0039"
)

image_path = folder / "C0039_1611.png"
label_path = folder / "ground_truth.txt"

# 이미지 불러오기
image = Image.open(image_path).convert("RGB")
draw = ImageDraw.Draw(image)

# 바운딩박스 찾기
with open(label_path, "r") as f:
    for line in f:
        values = list(map(float, line.split()))

        if int(values[0]) == 1611:
            _, ymin, ymax, xmin, xmax = values

            # 결함 위치 표시
            draw.rectangle(
                [(xmin, ymin), (xmax, ymax)],
                outline="red",
                width=3
            )

# 결과 이미지 출력
plt.figure(figsize=(6, 6))
plt.imshow(image)
plt.title("C0039_1611 - Ground Truth")
plt.axis("off")
plt.show()
